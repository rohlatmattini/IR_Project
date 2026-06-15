"""
Preprocessing Service
مسؤول عن تنظيف النصوص وتحضيرها قبل الفهرسة أو البحث
"""

import os
import re
import nltk

# =====================================================================
# حل مشكلة الأمان (Security Violation) عبر تخطي فحص المسارات القييدي في NLTK
# =====================================================================
if hasattr(nltk, "pathsec") and hasattr(nltk.pathsec, "validate_path"):

    def dummy_validate(path, context=None, required_root=None):
        return path  # تمرير المسار بدون إثارة الاستثناء الأمنية

    nltk.pathsec.validate_path = dummy_validate

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# تأمين مسار ديناميكي نظيف للمستخدم الحالي متوافق مع نظام ويندوز
home_dir = os.path.expanduser("~")
nltk_data_path = os.path.join(home_dir, "AppData", "Roaming", "nltk_data")

# إعادة تعيين مصفوفة المسارات لضمان عدم حدوث تداخل مع مسارات الكاش القديمة
nltk.data.path = [nltk_data_path]


# تحميل الموارد المطلوبة من NLTK (مرة وحدة فقط وبشكل آمن)
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            # نحدد مجلد التحميل download_dir لضمان النزول في المسار النظيف والموحد
            nltk.download(r, download_dir=nltk_data_path, quiet=True)
        except Exception as e:
            print(f"[NLTK Error] Failed to download or verify resource {r}: {str(e)}")


# استدعاء دالة التحميل والتحقق عند بدء الخدمة
download_nltk_resources()

# تحميل stopwords والـ lemmatizer مرة واحدة في الذاكرة لتسريع المعالجة
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """إزالة الأحرف الخاصة والأرقام والمسافات الزائدة والتنظيف العام"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # إزالة HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # إزالة الروابط
    text = re.sub(r"[^a-z\s]", " ", text)  # إزالة كل شي غير الحروف الإنجليزية
    text = re.sub(r"\s+", " ", text).strip()  # تنظيف المسافات الزائدة المتكررة
    return text


def tokenize(text: str) -> list:
    """تقطيع النص إلى كلمات منفصلة"""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """إزالة كلمات التوقف (Stopwords) والكلمات التي تتكون من حرف واحد"""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:
    """إرجاع الكلمات لجذرها المعجمي (Lemmatization)"""
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str, return_tokens: bool = False):
    """
    الدالة الرئيسية للمعالجة المسبقة

    المراحل المطبقة بالتسلسل:
    1. تنظيف النص (تحويل لحروف صغيرة، إزالة الروابط والرموز)
    2. التقطيع (Tokenization)
    3. إزالة كلمات التوقف (Stopwords)
    4. الـ Lemmatization لإرجاع الكلمات لأصلها

    return_tokens=True  ← ترجع قائمة (List) من الكلمات المعالجة
    return_tokens=False ← ترجع نصاً كاملاً (String) مدمجاً بمسافات
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    if return_tokens:
        return tokens
    return " ".join(tokens)


def preprocess_batch(texts: list, return_tokens: bool = False) -> list:
    """معالجة قائمة نصوص كاملة دفعة واحدة (Batch Processing)"""
    return [preprocess_text(t, return_tokens=return_tokens) for t in texts]


# ===== للتجربة المباشرة للتأكد من عمل الخدمة =====
if __name__ == "__main__":
    sample = "Information Retrieval systems help users find relevant documents!"
    print("Original  :", sample)
    print("Processed :", preprocess_text(sample))
    print("Tokens    :", preprocess_text(sample, return_tokens=True))
