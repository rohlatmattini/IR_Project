import os
import re
import warnings
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*Security Violation.*"
)

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)


def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]
    for r in resources:
        try:
            nltk.download(r, download_dir=nltk_data_path, quiet=True)
        except Exception as e:
            print(f"[NLTK Error] Failed to download or verify resource {r}: {str(e)}")


download_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:

    tagged_tokens = nltk.pos_tag(tokens)
    cleaned_lemmas = []

    for word, tag in tagged_tokens:
        if tag.startswith("V"):
            pos_attr = wordnet.VERB
        elif tag.startswith("J"):
            pos_attr = wordnet.ADJ
        elif tag.startswith("R"):
            pos_attr = wordnet.ADV
        else:
            pos_attr = wordnet.NOUN

        cleaned_lemmas.append(lemmatizer.lemmatize(word, pos=pos_attr))
    return cleaned_lemmas


def stem(tokens: list) -> list:
    return [stemmer.stem(t) for t in tokens]


def preprocess_text(text: str, return_tokens: bool = False, use_stemming: bool = False):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    if use_stemming:
        tokens = stem(tokens)

    if return_tokens:
        return tokens
    return " ".join(tokens)


def preprocess_batch(
    texts: list, return_tokens: bool = False, use_stemming: bool = False
) -> list:
    return [
        preprocess_text(t, return_tokens=return_tokens, use_stemming=use_stemming)
        for t in texts
    ]


if __name__ == "__main__":
    sample = "💡 Check out <p>The <b>cats</b> discovered a web link!</p> https://google.com They are chasing mice responsibly."

    print("=================== Text Preprocessing Pipeline ===================")
    print(f"[0] Original Text:\n    '{sample}'\n")

    cleaned = clean_text(sample)
    print(
        f"[1] After Cleaning (Removed HTML, Links, Punctuation, Lowercased):\n    '{cleaned}'\n"
    )

    tokens = tokenize(cleaned)
    print(
        f"[2] After Tokenization (Splitting text into a list of words):\n    {tokens}\n"
    )

    no_stopwords = remove_stopwords(tokens)
    print(
        f"[3] After Stopwords Removal (Removed short words & stopwords):\n    {no_stopwords}\n"
    )

    lemmatized = lemmatize(no_stopwords)
    print(
        f"[4] After Lemmatization (Dictionary-based root reduction: cats->cat, mice->mouse):\n    {lemmatized}\n"
    )

    stemmed = stem(no_stopwords)
    print(f"[5] After Stemming (Rule-based suffix stripping):\n    {stemmed}\n")
    print("===================================================================")
