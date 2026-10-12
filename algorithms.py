import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import pickle
import hashlib
import json
from parsivar import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow import string, constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


df = pd.read_csv(r'data/digikala_150k.csv')

normalizer = Normalizer()  # استفاده از parsivar

def clean_text(text):
    # نرمال‌سازی پیشرفته
    text = normalizer.normalize(text)

    # حذف کشیدگی حروف (عاااالی → عالی)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # حذف لینک
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # حذف ایموجی و کاراکترهای غیر فارسی
    text = re.sub(r'[^\u0600-\u06FF0-9\s]', ' ', text)

    # حذف فاصله‌های اضافی
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Remove invalid rows before splitting: missing/invalid labels, empty cleaned
# comments, and duplicate comments. Conflicting labels for identical comments
# are removed entirely to avoid assigning an arbitrary sentiment.
df = df.dropna(subset=['body', 'label']).copy()
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df[df['label'].isin([0, 1])].copy()
df['clean_body'] = df['body'].astype(str).apply(clean_text)
df = df[df['clean_body'].ne('')].copy()
conflicting_comment = df.groupby('clean_body')['label'].transform('nunique').gt(1)
df = df.loc[~conflicting_comment]
df = df.drop_duplicates(subset='clean_body', keep='first').reset_index(drop=True)
df['label'] = df['label'].astype(int)

# برچسب‌ها
texts = df['clean_body']
labels = df['label'].values

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# All classical models use the same TF-IDF settings, fitted only on Train.
TFIDF_CONFIG = {
    'max_features': 100_000,
    'ngram_range': (1, 2),
    'sublinear_tf': True,
}


def data_fingerprint(texts, labels):
    """Stable fingerprint of the ordered text/label rows used by a split."""
    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        encoded_text = str(text).encode('utf-8')
        digest.update(len(encoded_text).to_bytes(8, byteorder='big'))
        digest.update(encoded_text)
        digest.update(str(int(label)).encode('ascii'))
        digest.update(b'\n')
    return digest.hexdigest()


TRAIN_FINGERPRINT = data_fingerprint(X_train_text, y_train)
VALIDATION_FINGERPRINT = data_fingerprint(X_val_text, y_val)


def model_artifact_metadata(algorithm, vectorizer, estimator):
    return {
        'format_version': 2,
        'algorithm': algorithm,
        'train_fingerprint': TRAIN_FINGERPRINT,
        'validation_fingerprint': VALIDATION_FINGERPRINT,
        'split': {'train': 0.8, 'validation': 0.1, 'test': 0.1, 'random_state': 42},
        'tfidf_config': vectorizer.get_params(deep=False),
        'estimator_class': type(estimator).__name__,
        'estimator_params': estimator.get_params(deep=False),
    }


# پارامترها

# ==================== BiLSTM ALGORITHM ====================
# Adapt the text vocabulary on Train only, then save split provenance with the model.
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    ngrams=(1, 2)
)
vectorizer.adapt(X_train_text)

input_layer = layers.Input(shape=(1,), dtype=string)
vectorized_text = vectorizer(input_layer)
embedded_text = layers.Embedding(
    input_dim=MAX_VOCAB_SIZE,
    output_dim=64
)(vectorized_text)
bidirectional_lstm = layers.Bidirectional(
    layers.LSTM(
        units=48,
        return_sequences=True,
        dropout=0.4,
        recurrent_dropout=0.3
    )
)(embedded_text)
pooled_output = layers.GlobalMaxPooling1D()(bidirectional_lstm)
dropout_1 = layers.Dropout(0.5)(pooled_output)
output_layer = layers.Dense(units=1, activation='sigmoid')(dropout_1)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
    ModelCheckpoint('final_best_lstm_model.keras', monitor='val_loss', save_best_only=True,
                    mode='min', verbose=1)
]
X_train_lstm = np.asarray(X_train_text.astype(object), dtype=object).reshape(-1, 1)
X_val_lstm = np.asarray(X_val_text.astype(object), dtype=object).reshape(-1, 1)
history = model.fit(
    constant(X_train_lstm, dtype=string), y_train,
    validation_data=(constant(X_val_lstm, dtype=string), y_val),
    epochs=30, batch_size=512, callbacks=callbacks
)

LSTM_MODEL_PATH = 'final_best_lstm_model.keras'
LSTM_METADATA_PATH = 'final_best_lstm_model.metadata.json'
lstm_model = load_model(LSTM_MODEL_PATH)
model_digest = hashlib.sha256()
with open(LSTM_MODEL_PATH, 'rb') as lstm_file:
    for chunk in iter(lambda: lstm_file.read(1024 * 1024), b''):
        model_digest.update(chunk)
with open(LSTM_METADATA_PATH, 'w', encoding='utf-8') as metadata_file:
    json.dump({
        'format_version': 1,
        'algorithm': 'BiLSTM',
        'train_fingerprint': TRAIN_FINGERPRINT,
        'validation_fingerprint': VALIDATION_FINGERPRINT,
        'split': {'train': 0.8, 'validation': 0.1, 'test': 0.1, 'random_state': 42},
        'model_sha256': model_digest.hexdigest(),
    }, metadata_file, indent=2)


# ==================== SVM ALGORITHM ====================
# Fit vocabulary/IDF on Train only; transform Validation with the fitted vectorizer.
from sklearn.svm import LinearSVC

svm_vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
svm_X_train = svm_vectorizer.fit_transform(X_train_text)
svm_X_val = svm_vectorizer.transform(X_val_text)
svm_model = LinearSVC(class_weight='balanced', random_state=42)
svm_model.fit(svm_X_train, y_train)
svm_val_predictions = svm_model.predict(svm_X_val)

print('\nSVM validation results:')
print(classification_report(y_val, svm_val_predictions, digits=4))

with open('svm_sentiment_model.pkl', 'wb') as svm_file:
    pickle.dump({
        'vectorizer': svm_vectorizer,
        'model': svm_model,
        'metadata': model_artifact_metadata('SVM', svm_vectorizer, svm_model),
    }, svm_file)



# ==================== LOGISTIC REGRESSION ALGORITHM ====================
# Independent vectorizer; fit only on Train, then predict the shared Validation Set.
logistic_regression_vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
lr_X_train = logistic_regression_vectorizer.fit_transform(X_train_text)
lr_X_val = logistic_regression_vectorizer.transform(X_val_text)
logistic_regression_model = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42
)
logistic_regression_model.fit(lr_X_train, y_train)
logistic_regression_val_predictions = logistic_regression_model.predict(lr_X_val)

print('\nLogistic Regression validation results:')
print(classification_report(y_val, logistic_regression_val_predictions, digits=4))

with open('logistic_regression_sentiment_model.pkl', 'wb') as lr_file:
    pickle.dump({
        'vectorizer': logistic_regression_vectorizer,
        'model': logistic_regression_model,
        'metadata': model_artifact_metadata(
            'Logistic Regression', logistic_regression_vectorizer,
            logistic_regression_model
        ),
    }, lr_file)


# ==================== RANDOM FOREST ALGORITHM ====================
# Independent vectorizer; same configuration and Train-only fit as other models.
from sklearn.ensemble import RandomForestClassifier

random_forest_vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
rf_X_train = random_forest_vectorizer.fit_transform(X_train_text)
rf_X_val = random_forest_vectorizer.transform(X_val_text)
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
random_forest_model.fit(rf_X_train, y_train)
random_forest_val_predictions = random_forest_model.predict(rf_X_val)

print('\nRandom Forest validation results:')
print(classification_report(y_val, random_forest_val_predictions, digits=4))

with open('random_forest_sentiment_model.pkl', 'wb') as rf_file:
    pickle.dump(
        {
            'vectorizer': random_forest_vectorizer,
            'model': random_forest_model,
            'metadata': model_artifact_metadata(
                'Random Forest', random_forest_vectorizer, random_forest_model
            ),
        },
        rf_file
    )


# ==================== NAIVE BAYES ALGORITHM ====================
# Independent vectorizer; same configuration and Train-only fit as other models.
naive_bayes_vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
nb_X_train = naive_bayes_vectorizer.fit_transform(X_train_text)
nb_X_val = naive_bayes_vectorizer.transform(X_val_text)

naive_bayes_model = MultinomialNB(alpha=0.5)
naive_bayes_model.fit(nb_X_train, y_train)
naive_bayes_val_predictions = naive_bayes_model.predict(nb_X_val)

print('\nNaive Bayes validation results:')
print(classification_report(y_val, naive_bayes_val_predictions, digits=4))

with open('naive_bayes_sentiment_model.pkl', 'wb') as nb_file:
    pickle.dump(
        {
            'vectorizer': naive_bayes_vectorizer,
            'model': naive_bayes_model,
            'metadata': model_artifact_metadata(
                'Naive Bayes', naive_bayes_vectorizer, naive_bayes_model
            ),
        },
        nb_file
    )


# ==================== SAVE MODEL ACCURACY HISTORY ====================
# Evaluate the saved BiLSTM checkpoint and save one validation accuracy per model.
lstm_comparison_model = load_model('final_best_lstm_model.keras')
lstm_val_probabilities = lstm_comparison_model.predict(
    constant(X_val_lstm, dtype=string), verbose=0
).reshape(-1)
lstm_val_predictions = (lstm_val_probabilities >= 0.5).astype(int)

def validation_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
    }


accuracy_history = pd.DataFrame([
    {'Algorithm': 'BiLSTM', **validation_metrics(y_val, lstm_val_predictions)},
    {'Algorithm': 'SVM', **validation_metrics(y_val, svm_val_predictions)},
    {'Algorithm': 'Logistic Regression', **validation_metrics(y_val, logistic_regression_val_predictions)},
    {'Algorithm': 'Naive Bayes', **validation_metrics(y_val, naive_bayes_val_predictions)},
    {'Algorithm': 'Random Forest', **validation_metrics(y_val, random_forest_val_predictions)},
])
accuracy_history.to_csv('model_accuracy_history.csv', index=False)
print('Saved model validation accuracies to model_accuracy_history.csv')
