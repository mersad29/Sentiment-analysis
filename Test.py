"""Interactively test new Persian reviews with the saved Logistic Regression model."""

import pickle
import re
from pathlib import Path

from parsivar import Normalizer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "logistic_regression_sentiment_model.pkl"
normalizer = Normalizer()

# نظرهای مورد آزمایش را اینجا اضافه یا جایگزین کنید.
TEST_COMMENTS = [
    'کیفیت صداش واقعاً عالیه، بیس خوبی داره و باتریش هم بیشتر از چیزی که فکر می‌کردم دوام میاره. کاملاً راضی‌ام.',
    'عااااالی بوددد 😍 هم سبکه هم خیلی راحته، انگار اصلاً کفش پام نیست!',
    'نسبت به قیمتش واقعاً ارزش خرید داره. نوشیدنی رو چند ساعت گرم نگه می‌داره و نشتی هم نداره. پیشنهادش می‌کنم.',
    'دوربینش فوق‌العاده‌ست 📸 عکس‌ها حتی توی نور کم هم کیفیت خوبی دارن. از خریدم خیلی خوشحالم.',
    'بافت سبکی داره و سریع جذب پوست می‌شه. تا الان بهترین گزینه‌ای بوده که برای استفاده روزانه امتحان کردم.',
    'Honestly، از چیزی که انتظار داشتم بهتره! کلیدها نرم هستن و تایپ کردن باهاش خیلی لذت‌بخشه.',
    'دو ماهه دارم ازش استفاده می‌کنم و هنوز مثل روز اوله. جاداره، زیپ‌هاش محکمه و دوختش هم تمیزه.',
    'واااای چه قهوه‌ای درست می‌کنه ☕ کار کردن باهاش آسونه و واقعاً صبح‌ها رو برام لذت‌بخش‌تر کرده.',
    'نورش قابل تنظیمه و چشم رو اذیت نمی‌کنه. طراحی ساده و قشنگی هم داره. خرید خوبی بود.',
    'دمتـون گرم، بالاخره یه پاوربانک درست‌وحسابی پیدا کردم! سریع شارژ می‌کنه، ظرفیتش خوبه و کیفیت ساختش هم عالیه 👌'
    'اصلاً راضی نیستم. مدام اتصالش قطع می‌شه و کیفیت میکروفونش هم خیلی ضعیفه. پولم رو دور ریختم.',
    'افتضاااااحه 😡 بعد از دو بار استفاده بوی سوختگی داد. واقعاً انتظار همچین کیفیتی رو نداشتم.',
    'جنس پارچه خیلی نازک‌تر از عکس بود. بعد از اولین شست‌وشو هم تغییر شکل داد. به هیچ وجه توصیه نمی‌کنم.',
    'باتریش افتضاحه، هر روز باید شارژش کنم. تازه بعضی وقت‌ها هم قدم‌ها رو اشتباه می‌شمره 🙄',
    'بعد از نیم ساعت نشستن کمرم درد می‌گیره. صندلی سفت و ناراحتیه و تنظیم ارتفاعش هم درست کار نمی‌کنه.',
    'The performance is terrible. حتی برای کارهای ساده هم هنگ می‌کنه و فنش صدای زیادی داره. اصلاً ارزش این قیمت رو نداره.',
    'رنگی که فرستادن با عکس فرق داشت. لبه‌هاش هم خیلی زود شکست. کیفیتش واقعاً ناامیدکننده‌ست.',
    'عجب خرید بدی کردم! 😤 صدای خیلی زیادی داره و قدرت مکشش هم بعد از چند دقیقه کم می‌شه. اصلاً به درد نمی‌خوره.',
    'ماندگاریش تقریباً صفره. نیم ساعت بعد هیچ بویی ازش نمی‌مونه. با این قیمت انتظار خیلی بیشتری داشتم.',
    'مسخره‌ست واقعاً! هر بار یه عدد متفاوت نشون می‌ده، حتی وقتی چند ثانیه فاصله بین اندازه‌گیری‌هاست. اصلاً قابل اعتماد نیست.'
]


def load_saved_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved Logistic Regression model was not found: {MODEL_PATH}\n"
            "Run algorithms.py first to train and save the model."
        )

    with MODEL_PATH.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if not isinstance(artifact, dict) or not {"model", "vectorizer"}.issubset(artifact):
        raise ValueError("The pickle must contain the saved Logistic Regression model and vectorizer.")

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]
    if type(model).__name__ != "LogisticRegression":
        raise ValueError(f"Expected LogisticRegression, found {type(model).__name__}.")
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("The saved TF-IDF vectorizer is not fitted.")

    metadata = artifact.get("metadata")
    if metadata and metadata.get("algorithm") != "Logistic Regression":
        raise ValueError("This pickle is not marked as a Logistic Regression artifact.")
    return vectorizer, model


vectorizer, model = load_saved_model()


def clean_text(text):
    """Apply the same Persian normalization and cleaning used during training."""
    text = normalizer.normalize(str(text))
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_sentiment(text):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("نظر پس از پاک‌سازی خالی است؛ لطفاً متن دیگری وارد کنید.")

    features = vectorizer.transform([cleaned])
    predicted_label = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    class_to_probability = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    confidence = class_to_probability[predicted_label]

    return {
        "text": text,
        "clean_text": cleaned,
        "label": predicted_label,
        "sentiment": "مثبت" if predicted_label == 1 else "منفی",
        "confidence": confidence,
        "positive_probability": class_to_probability.get(1, 0.0),
    }


def main():
    print("پیش‌بینی نظرهای تعریف‌شده در TEST_COMMENTS با Logistic Regression:\n")
    for index, text in enumerate(TEST_COMMENTS, start=1):
        try:
            result = predict_sentiment(text)
        except ValueError as error:
            print(f"نمونهٔ {index}: {error}")
            continue

        print(f"نمونهٔ {index}: {result['text']}")
        print(f"پیش‌بینی: {result['sentiment']} (برچسب {result['label']})")
        print(f"اطمینان پیش‌بینی: {result['confidence']:.2%}")
        print(f"احتمال مثبت بودن: {result['positive_probability']:.2%}")
        print("-" * 50)


if __name__ == "__main__":
    main()
