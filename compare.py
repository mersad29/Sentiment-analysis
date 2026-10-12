"""Compare saved TF-IDF sentiment models and chart validation metrics.

Run from the project root with: python compare.py
"""

import re
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parsivar import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tensorflow import constant, string


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "digikala_150k.csv"
LSTM_MODEL_FILE = ROOT / "final_best_lstm_model.keras"
LSTM_METADATA_FILE = ROOT / "final_best_lstm_model.metadata.json"
MODEL_FILES = {
    "SVM": ROOT / "svm_sentiment_model.pkl",
    "Logistic Regression": ROOT / "logistic_regression_sentiment_model.pkl",
    "Random Forest": ROOT / "random_forest_sentiment_model.pkl",
    "Naive Bayes": ROOT / "naive_bayes_sentiment_model.pkl",
}
OUTPUT_CSV = ROOT / "model_metrics_comparison.csv"
OUTPUT_CHART = ROOT / "assets" / "algorithm_metrics_comparison.png"
REPORTS_JSON = ROOT / "classification_reports.json"
CONFUSION_DIR = ROOT / "confusion_matrices"
CONFUSION_MODELS = {"Logistic Regression", "BiLSTM"}
TFIDF_CONFIG = {
    "max_features": 100_000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}
MODEL_CLASSES = {
    "SVM": "LinearSVC",
    "Logistic Regression": "LogisticRegression",
    "Random Forest": "RandomForestClassifier",
    "Naive Bayes": "MultinomialNB",
}


def clean_text(text, normalizer):
    text = normalizer.normalize(str(text))
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def data_fingerprint(texts, labels):
    """Match the ordered split fingerprint written by algorithms.py."""
    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        encoded_text = str(text).encode("utf-8")
        digest.update(len(encoded_text).to_bytes(8, byteorder="big"))
        digest.update(encoded_text)
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_and_show_confusion_matrix(matrix, model_name, output_path):
    """Render a confusion matrix from already-loaded model predictions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Negative", "Positive"])
    ax.set_yticks([0, 1], labels=["Negative", "Positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} Confusion Matrix (Validation Set)")
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            ax.text(
                column, row, f"{value:,}", ha="center", va="center",
                color="white" if value > midpoint else "black", fontsize=12
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    print(f"Saved {model_name} confusion matrix image to {output_path}")
    plt.show()
    plt.close(fig)


def validate_artifact(name, artifact, expected_vectorizer, train_fingerprint, val_fingerprint):
    """Reject legacy, stale, or inconsistent model/vectorizer pickle artifacts."""
    if not isinstance(artifact, dict) or not {"model", "vectorizer", "metadata"}.issubset(artifact):
        raise ValueError("legacy artifact format; it has no training/split metadata")

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]
    metadata = artifact["metadata"]
    expected_split = {"train": 0.8, "validation": 0.1, "test": 0.1, "random_state": 42}

    checks = {
        "format version": metadata.get("format_version") == 2,
        "algorithm name": metadata.get("algorithm") == name,
        "train split fingerprint": metadata.get("train_fingerprint") == train_fingerprint,
        "validation split fingerprint": metadata.get("validation_fingerprint") == val_fingerprint,
        "split settings": metadata.get("split") == expected_split,
        "TF-IDF settings": metadata.get("tfidf_config") == expected_vectorizer.get_params(deep=False),
        "model class": type(model).__name__ == MODEL_CLASSES[name]
            and metadata.get("estimator_class") == MODEL_CLASSES[name],
        "model parameters": metadata.get("estimator_params") == model.get_params(deep=False),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise ValueError(f"incompatible artifact ({failed})")

    # Metadata alone is not enough: verify that the saved vocabulary and IDF
    # really came from this exact training split.
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise ValueError("vectorizer is not fitted")
    if vectorizer.get_params(deep=False) != expected_vectorizer.get_params(deep=False):
        raise ValueError("saved vectorizer settings do not match the expected TF-IDF settings")
    if vectorizer.vocabulary_ != expected_vectorizer.vocabulary_ or not np.array_equal(
        vectorizer.idf_, expected_vectorizer.idf_
    ):
        raise ValueError("saved vectorizer was not fitted on the current Train split")
    if getattr(model, "n_features_in_", None) != len(vectorizer.vocabulary_):
        raise ValueError("model feature count does not match its saved vectorizer")
    if not np.array_equal(getattr(model, "classes_", []), np.array([0, 1])):
        raise ValueError("model classes do not match expected sentiment labels 0 and 1")

    return vectorizer, model


def main():
    required_files = [DATA_PATH, LSTM_MODEL_FILE, LSTM_METADATA_FILE, *MODEL_FILES.values()]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(f"Required data/model files are missing:\n{missing}\nRun algorithms.py first.")

    frame = pd.read_csv(DATA_PATH).dropna(subset=["body", "label"]).copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[frame["label"].isin([0, 1])].copy()
    normalizer = Normalizer()
    frame["clean_body"] = frame["body"].map(lambda text: clean_text(text, normalizer))
    frame = frame[frame["clean_body"].ne("")].copy()
    conflicting_comment = frame.groupby("clean_body")["label"].transform("nunique").gt(1)
    frame = frame.loc[~conflicting_comment]
    frame = frame.drop_duplicates(subset="clean_body", keep="first").reset_index(drop=True)
    texts = frame["clean_body"]
    labels = frame["label"].astype(int)

    # Reproduce the same stratified 80/10/10 split used in algorithms.py.
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=42,
        stratify=y_temp
    )

    train_fingerprint = data_fingerprint(X_train_text, y_train)
    val_fingerprint = data_fingerprint(X_val_text, y_val)
    expected_vectorizer = TfidfVectorizer(**TFIDF_CONFIG).fit(X_train_text)

    # Require provenance for the LSTM too; the old .keras checkpoint alone
    # cannot prove it was trained against this exact split.
    try:
        lstm_metadata = json.loads(LSTM_METADATA_FILE.read_text(encoding="utf-8"))
        expected_split = {"train": 0.8, "validation": 0.1, "test": 0.1, "random_state": 42}
        if (
            lstm_metadata.get("format_version") != 1
            or lstm_metadata.get("algorithm") != "BiLSTM"
            or lstm_metadata.get("train_fingerprint") != train_fingerprint
            or lstm_metadata.get("validation_fingerprint") != val_fingerprint
            or lstm_metadata.get("split") != expected_split
            or lstm_metadata.get("model_sha256") != file_sha256(LSTM_MODEL_FILE)
        ):
            raise ValueError("LSTM checkpoint metadata does not match the current split/model file")
        from tensorflow.keras.models import load_model
        lstm_model = load_model(LSTM_MODEL_FILE)
        if not isinstance(lstm_model.output_shape, tuple) or lstm_model.output_shape[-1] != 1:
            raise ValueError("LSTM checkpoint is not a binary sigmoid classifier")
    except Exception as error:
        raise ValueError(
            f"Refusing to evaluate {LSTM_MODEL_FILE.name}: {error}. "
            "Run the updated algorithms.py to train and save a compatible LSTM checkpoint."
        ) from error

    # Validate every artifact before predicting or writing any result files.
    # This prevents partial reports when a later model is stale/incompatible.
    loaded_models = {}
    for name, model_path in MODEL_FILES.items():
        try:
            with model_path.open("rb") as model_file:
                artifact = pickle.load(model_file)
            vectorizer, model = validate_artifact(
                name, artifact, expected_vectorizer,
                train_fingerprint, val_fingerprint
            )
        except Exception as error:
            raise ValueError(
                f"Refusing to evaluate {model_path.name}: {error}. "
                "Run the updated algorithms.py to retrain and save compatible artifacts."
            ) from error
        loaded_models[name] = (vectorizer, model)

    predictions_by_model = {}
    for name, (vectorizer, model) in loaded_models.items():
        validation_features = vectorizer.transform(X_val_text)
        predictions_by_model[name] = model.predict(validation_features)

    lstm_validation_text = np.asarray(
        X_val_text.astype(object), dtype=object
    ).reshape(-1, 1)
    lstm_probabilities = np.asarray(
        lstm_model.predict(constant(lstm_validation_text, dtype=string), verbose=0)
    ).reshape(-1)
    if len(lstm_probabilities) != len(y_val) or not np.isfinite(lstm_probabilities).all():
        raise ValueError("LSTM returned invalid probabilities for the shared validation set")
    predictions_by_model["BiLSTM"] = (lstm_probabilities >= 0.5).astype(int)

    results = []
    reports = {}
    for name, predictions in predictions_by_model.items():
        report = classification_report(
            y_val, predictions, labels=[0, 1], target_names=["Negative", "Positive"],
            output_dict=True, zero_division=0
        )
        reports[name] = report

        print(f"\n{name} Classification Report:")
        print(classification_report(
            y_val, predictions, labels=[0, 1],
            target_names=["Negative", "Positive"], zero_division=0
        ))

        if name in CONFUSION_MODELS:
            matrix = confusion_matrix(y_val, predictions, labels=[0, 1])
            matrix_frame = pd.DataFrame(
                matrix,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )
            matrix_path = CONFUSION_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv"
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_frame.to_csv(matrix_path)
            print(f"{name} Confusion Matrix (saved to {matrix_path}):")
            print(matrix_frame.to_string())

            image_name = "lstm_confusion_matrix.png" if name == "BiLSTM" else "logistic_regression_confusion_matrix.png"
            save_and_show_confusion_matrix(
                matrix, name, ROOT / "assets" / image_name
            )

        results.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro Precision": precision_score(y_val, predictions, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_val, predictions, average="macro", zero_division=0),
            "Macro F1": f1_score(y_val, predictions, average="macro", zero_division=0),
        })

    metrics = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    metrics.to_csv(OUTPUT_CSV, index=False)
    REPORTS_JSON.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Validation metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved summary metrics to {OUTPUT_CSV}")
    print(f"Saved classification reports to {REPORTS_JSON}")

    chart_metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    chart_data = metrics.set_index("Algorithm")[chart_metrics]
    ax = chart_data.plot(kind="bar", figsize=(12, 7), width=0.82)
    # Zoom the vertical axis around the observed scores so small differences
    # are visible; labels preserve the exact values and the truncated baseline
    # is made explicit in the axis title.
    score_min = float(chart_data.min().min())
    score_max = float(chart_data.max().max())
    score_span = score_max - score_min
    padding = max(score_span * 0.30, 0.005)
    lower_bound = max(0.0, score_min - padding)
    upper_bound = min(1.0, score_max + padding)
    if upper_bound - lower_bound < 0.01:
        lower_bound = max(0.0, score_min - 0.01)
        upper_bound = min(1.0, score_max + 0.01)
    ax.set_ylim(lower_bound, upper_bound)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, rotation=90)
    ax.set_title("Validation Metrics on the Shared Validation Set")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(f"Score (zoomed axis: {lower_bound:.3f}–{upper_bound:.3f})")
    ax.legend(title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150)
    print(f"Saved chart to {OUTPUT_CHART}")
    plt.show()


if __name__ == "__main__":
    main()
