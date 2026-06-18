
import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)


def download_nltk_resources():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
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
    return [lemmatizer.lemmatize(t) for t in tokens]


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


def preprocess_batch(texts: list, return_tokens: bool = False, use_stemming: bool = False) -> list:
    return [
        preprocess_text(t, return_tokens=return_tokens, use_stemming=use_stemming)
        for t in texts
    ]


if __name__ == "__main__":
    sample = "Information Retrieval systems help users find relevant documents!"
    print("Original          :", sample)
    print("Processed (lemma)  :", preprocess_text(sample))
    print("Processed (stem)   :", preprocess_text(sample, use_stemming=True))
    print("Tokens (lemma)      :", preprocess_text(sample, return_tokens=True))
    print("Tokens (stem)       :", preprocess_text(sample, return_tokens=True, use_stemming=True))