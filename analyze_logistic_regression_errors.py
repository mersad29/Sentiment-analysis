"""Analyze Logistic Regression errors on the project's fixed Test Set.

This script loads the saved classifier and vectorizer. It does not retrain the
sentiment model. Run from the project root with:
    python analyze_logistic_regression_errors.py
"""

import hashlib
import json
import pickle
import re
from pathlib import Path

import pandas as pd
from parsivar import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "digikala_150k.csv"
MODEL_PATH = ROOT / "logistic_regression_sentiment_model.pkl"
ERRORS_CSV = ROOT / "logistic_regression_misclassified_test.csv"
CLUSTERS_CSV = ROOT / "logistic_regression_error_clusters.csv"
SUMMARY_JSON = ROOT / "logistic_regression_error_summary.json"

TFIDF_CONFIG = {
    "max_features": 100_000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}
SPLIT_CONFIG = {"train": 0.8, "validation": 0.1, "test": 0.1, "random_state": 42}
LABEL_NAMES = {0: "Negative", 1: "Positive"}
ERROR_TYPE_NAMES = {
    "False Positive": "negative review predicted as positive",
    "False Negative": "positive review predicted as negative",
}


def clean_text(text, normalizer):
    text = normalizer.normalize(str(text))
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def data_fingerprint(texts, labels):
    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        encoded_text = str(text).encode("utf-8")
        digest.update(len(encoded_text).to_bytes(8, byteorder="big"))
        digest.update(encoded_text)
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_test_split():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH).dropna(subset=["body", "label"]).copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[frame["label"].isin([0, 1])].copy()
    normalizer = Normalizer()
    frame["clean_body"] = frame["body"].map(lambda text: clean_text(text, normalizer))
    frame = frame[frame["clean_body"].ne("")].copy()
    conflicting = frame.groupby("clean_body")["label"].transform("nunique").gt(1)
    frame = frame.loc[~conflicting]
    frame = frame.drop_duplicates(subset="clean_body", keep="first").reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)

    # Keep this two-stage split identical to algorithms.py and compare.py.
    X_train, X_temp, y_train, y_temp = train_test_split(
        frame["clean_body"], frame["label"].to_numpy(),
        test_size=0.2, random_state=42, stratify=frame["label"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return frame, X_train, y_train, X_val, y_val, X_test, y_test


def load_compatible_model(train_text, train_labels, val_text, val_labels):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved Logistic Regression model not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict) or not {"model", "vectorizer", "metadata"}.issubset(artifact):
        raise ValueError(
            "The saved Logistic Regression pickle is a legacy artifact without split metadata. "
            "It cannot be safely matched to the current Test Set."
        )

    model, vectorizer, metadata = artifact["model"], artifact["vectorizer"], artifact["metadata"]
    checks = {
        "format version": metadata.get("format_version") == 2,
        "algorithm": metadata.get("algorithm") == "Logistic Regression",
        "Train split": metadata.get("train_fingerprint") == data_fingerprint(train_text, train_labels),
        "Validation split": metadata.get("validation_fingerprint") == data_fingerprint(val_text, val_labels),
        "split configuration": metadata.get("split") == SPLIT_CONFIG,
        "TF-IDF configuration": metadata.get("tfidf_config") == TfidfVectorizer(**TFIDF_CONFIG).get_params(deep=False),
        "estimator class": type(model).__name__ == "LogisticRegression"
            and metadata.get("estimator_class") == "LogisticRegression",
        "estimator parameters": metadata.get("estimator_params") == model.get_params(deep=False),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Saved Logistic Regression artifact does not match the current data/split ("
            + ", ".join(failed)
            + "). Use a compatible saved model; this analysis will not retrain it."
        )
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise ValueError("The saved TF-IDF vectorizer is not fitted.")
    if getattr(model, "n_features_in_", None) != len(vectorizer.vocabulary_):
        raise ValueError("Saved model feature count does not match its vectorizer.")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError("Saved model classes are not the expected labels [0, 1].")
    return vectorizer, model


def cluster_errors(errors):
    """Group mistakes by FP/FN type and TF-IDF text-content similarity."""
    errors = errors.copy()
    errors["content_cluster"] = ""
    errors["cluster_keywords"] = ""
    cluster_summaries = []
    if errors.empty:
        return errors, cluster_summaries

    content_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, max_features=30_000,
        sublinear_tf=True
    )
    try:
        content_matrix = content_vectorizer.fit_transform(errors["clean_text"])
    except ValueError:
        errors["content_cluster"] = errors["error_type"] + "_01"
        return errors, cluster_summaries
    terms = content_vectorizer.get_feature_names_out()

    for error_type, group in errors.groupby("error_type", sort=True):
        row_positions = errors.index.get_indexer(group.index)
        group_matrix = content_matrix[row_positions]
        count = len(group)
        cluster_count = min(6, max(1, round((count / 10) ** 0.5)))
        if count < 2:
            cluster_count = 1

        if cluster_count == 1:
            assignments = [0] * count
            centroids = group_matrix.mean(axis=0)
        else:
            clusterer = MiniBatchKMeans(
                n_clusters=cluster_count, random_state=42, n_init=10,
                batch_size=min(1024, max(100, count))
            )
            assignments = clusterer.fit_predict(group_matrix)
            centroids = clusterer.cluster_centers_

        for cluster_id in sorted(set(assignments)):
            local_rows = [i for i, assigned in enumerate(assignments) if assigned == cluster_id]
            center = centroids[cluster_id] if cluster_count > 1 else centroids
            center = getattr(center, "A1", center).ravel()
            top_indices = center.argsort()[::-1]
            keywords = [str(terms[i]) for i in top_indices if center[i] > 0][:6]
            key = f"{error_type.replace(' ', '_')}_{cluster_id + 1:02d}"
            global_rows = group.index[local_rows]
            errors.loc[global_rows, "content_cluster"] = key
            errors.loc[global_rows, "cluster_keywords"] = ", ".join(keywords)
            cluster_summaries.append({
                "error_type": error_type,
                "error_meaning": ERROR_TYPE_NAMES[error_type],
                "content_cluster": key,
                "sample_count": len(local_rows),
                "keywords": keywords,
                "example_comments": errors.loc[global_rows, "comment"].head(3).tolist(),
            })
    return errors, cluster_summaries


def main():
    frame, X_train, y_train, X_val, y_val, X_test, y_test = load_test_split()
    vectorizer, model = load_compatible_model(X_train, y_train, X_val, y_val)

    # Only inference: the saved sentiment estimator and its vectorizer are loaded above.
    test_features = vectorizer.transform(X_test)
    predictions = model.predict(test_features).astype(int)
    mistakes = predictions != y_test
    test_count = len(y_test)
    mistake_count = int(mistakes.sum())

    test_frame = frame.loc[X_test.index].copy()
    error_frame = pd.DataFrame({
        "row_index": test_frame.index[mistakes],
        "comment": test_frame.loc[mistakes, "body"].astype(str).to_numpy(),
        "clean_text": X_test.loc[mistakes].to_numpy(),
        "actual_label": y_test[mistakes],
        "actual_label_name": [LABEL_NAMES[int(label)] for label in y_test[mistakes]],
        "predicted_label": predictions[mistakes],
        "predicted_label_name": [LABEL_NAMES[int(label)] for label in predictions[mistakes]],
        "error_type": [
            "False Positive" if actual == 0 and predicted == 1 else "False Negative"
            for actual, predicted in zip(y_test[mistakes], predictions[mistakes])
        ],
    })

    error_frame, cluster_summaries = cluster_errors(error_frame)
    error_frame.to_csv(ERRORS_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(cluster_summaries).to_csv(CLUSTERS_CSV, index=False, encoding="utf-8-sig")

    type_counts = error_frame["error_type"].value_counts().to_dict() if mistake_count else {}
    summary = {
        "model": "Logistic Regression",
        "evaluation_split": "Test",
        "test_count": test_count,
        "misclassified_count": mistake_count,
        "misclassified_percent": round(100 * mistake_count / test_count, 4) if test_count else 0.0,
        "test_accuracy": accuracy_score(y_test, predictions),
        "false_positive_count": int(type_counts.get("False Positive", 0)),
        "false_negative_count": int(type_counts.get("False Negative", 0)),
        "clusters_by_error_type_and_content": cluster_summaries,
        "error_csv": str(ERRORS_CSV),
        "cluster_csv": str(CLUSTERS_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Test samples: {test_count}")
    print(f"Misclassified: {mistake_count} ({summary['misclassified_percent']:.2f}%)")
    print(f"False positives: {summary['false_positive_count']}")
    print(f"False negatives: {summary['false_negative_count']}")
    print("\nMisclassified reviews:")
    for row in error_frame.itertuples(index=False):
        print(
            f"\nType: {row.error_type} | Actual: {row.actual_label_name} "
            f"({row.actual_label}) | Predicted: {row.predicted_label_name} "
            f"({row.predicted_label}) | Cluster: {row.content_cluster} "
            f"[{row.cluster_keywords}]\nComment: {row.comment}"
        )
    print(f"\nSaved misclassified samples: {ERRORS_CSV}")
    print(f"Saved content/error clusters: {CLUSTERS_CSV}")
    print(f"Saved summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
