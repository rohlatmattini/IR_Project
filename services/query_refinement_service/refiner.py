import os
import sys
import warnings

# sys.stdout.reconfigure(encoding="utf-8")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from collections import Counter
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


try:
    from spellchecker import SpellChecker

    spell = SpellChecker()
    SPELL_CHECK_AVAILABLE = True
except ImportError:
    SPELL_CHECK_AVAILABLE = False
    print(
        "[QueryRefinement] ⚠️ pyspellchecker is not available, install it using: pip install pyspellchecker"
    )


def spell_correct(query: str) -> str:
    if not SPELL_CHECK_AVAILABLE:
        return query
    words = query.split()
    corrected = []
    for word in words:
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return " ".join(corrected)


def get_synonyms(word: str, topn: int = 3) -> list:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name.lower())
    return list(synonyms)[:topn]


def expand_with_synonyms(query: str) -> str:
    """يضيف مرادفات فقط للكلمات المفيدة (طولها > 4 ومش stopwords)"""
    # كلمات لا نأخذ لها مرادفات
    SKIP_WORDS = {
        "should",
        "would",
        "could",
        "might",
        "must",
        "shall",
        "have",
        "has",
        "had",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "about",
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        "the",
        "and",
        "but",
        "for",
        "are",
        "was",
        "were",
    }

    words = query.split()
    expanded = list(words)
    topn = config.QUERY_REFINEMENT["synonym_topn"]

    for word in words:
        # نأخذ مرادفات فقط للكلمات المهمة
        if len(word) > 4 and word.lower() not in SKIP_WORDS:
            syns = get_synonyms(word, topn=topn)
            expanded.extend(syns)

    seen = set()
    result = []
    for w in expanded:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return " ".join(result)


def weight_with_history(query: str, history: list) -> str:
    if not history:
        return query
    max_history = config.QUERY_REFINEMENT["max_history"]
    recent_history = history[-max_history:]
    history_words = []
    for h_query in recent_history:
        history_words.extend(h_query.split())
    word_freq = Counter(history_words)
    top_history_words = [
        w for w, _ in word_freq.most_common(3) if w not in query.split()
    ]
    enriched = query + " " + " ".join(top_history_words)
    return enriched.strip()


try:
    from transformers import MarianMTModel, MarianTokenizer

    _translator = None
    _tokenizer = None
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print(
        "[Multilingual] ⚠️ transformers is not available, install it using: pip install transformers"
    )


def detect_language(text: str) -> str:
    """Detect text language based on Arabic character ratio"""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic_chars > len(text) * 0.3 else "en"


def translate_to_english(text: str) -> str:
    """Translate Arabic text to English using MarianMT directly"""
    global _translator, _tokenizer
    if not TRANSLATION_AVAILABLE:
        return text
    if _translator is None:
        print("[Multilingual] Loading translation model...")
        model_name = config.MULTILINGUAL["translation_model"]
        _tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translator = MarianMTModel.from_pretrained(model_name)
        print("[Multilingual] ✅ Model is ready")
    tokens = _tokenizer(
        [text], return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    translated = _translator.generate(**tokens)
    result = _tokenizer.decode(translated[0], skip_special_tokens=True)
    return result


# ضيف هاد الكود تحت دالة translate_to_english مباشرة، قبل دالة refine_query


def detect_and_translate(text: str) -> dict:
    """
    دالة مستقلة لتجربة Multilingual لحالها (بدون أي علاقة بـ synonyms/history).
    """
    lang = detect_language(text)
    result = {
        "original_text": text,
        "detected_language": lang,
        "translated_text": text,
        "translation_applied": False,
    }
    if lang == "ar":
        translated = translate_to_english(text)
        result["translated_text"] = translated
        result["translation_applied"] = True
    return result


def refine_query(
    query: str,
    history: list = None,
    use_synonyms: bool = None,
    use_spell: bool = None,
    use_history: bool = None,
) -> dict:
    """
    Main function for query refinement
    """
    history = history or []
    use_synonyms = (
        use_synonyms
        if use_synonyms is not None
        else config.QUERY_REFINEMENT["use_synonyms"]
    )
    use_spell = (
        use_spell
        if use_spell is not None
        else config.QUERY_REFINEMENT["use_spell_check"]
    )
    use_history = (
        use_history
        if use_history is not None
        else config.QUERY_REFINEMENT["use_history_weighting"]
    )

    result = {
        "original_query": query,
        "refined_query": query,
        "language": "en",
        "corrections": [],
        "synonyms_added": [],
    }

    lang = detect_language(query)
    result["language"] = lang
    if lang == "ar":
        query = translate_to_english(query)
        result["corrections"].append(f"Translated from Arabic: {query}")

    if use_spell:
        corrected = spell_correct(query)
        if corrected != query:
            result["corrections"].append(f"Correction: '{query}' -> '{corrected}'")
            query = corrected

    if use_history and history:
        query = weight_with_history(query, history)

    if use_synonyms:
        expanded = expand_with_synonyms(query)
        new_words = [w for w in expanded.split() if w not in query.split()]
        result["synonyms_added"] = new_words
        query = expanded

    result["refined_query"] = query
    return result


if __name__ == "__main__":
    print("=== Testing English Query ===")
    r = refine_query(
        "infomation retrival systm", history=["search engines", "document ranking"]
    )
    print("Refined:", r["refined_query"])
    print("Corrections:", r["corrections"])
    print("Synonyms:", r["synonyms_added"][:5])

    print("\n=== Testing Arabic Query ===")
    r2 = refine_query("استرجاع المعلومات")
    print("Original Query: استرجاع المعلومات")
    print("Translated & Refined:", r2["refined_query"])
    print("Language detected:", r2["language"])
