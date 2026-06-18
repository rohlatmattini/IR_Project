import os

# ===== المسارات =====
# صحيح:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset 2 (Webis Touché 2020) - الوحيد المستخدم
DATASET2_DIR = os.path.join(DATA_DIR, 'dataset2')
DATASET2_DOCS   = os.path.join(DATASET2_DIR, 'documents.json')
DATASET2_QUERIES = os.path.join(DATASET2_DIR, 'queries.json')
DATASET2_QRELS  = os.path.join(DATASET2_DIR, 'qrels.json')

# مجلدات الفهارس
INDEX_DIR = os.path.join(BASE_DIR, 'indexes')
INDEX2_DIR = os.path.join(INDEX_DIR, 'dataset2')

# مجلدات حفظ النماذج
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(INDEX2_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== إعدادات المعالجة المسبقة =====
PREPROCESSING = {
    'remove_stopwords': True,
    'stemming': False,
    'lemmatization': True,
    'lowercase': True,
    'remove_punctuation': True,
    'min_token_length': 2,
}

# ===== إعدادات BM25 =====
BM25_PARAMS = {
    'k1': 1.5,
    'b': 0.75,
}

# ===== إعدادات Embedding =====
EMBEDDING = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 64,
    'max_seq_length': 128,
}

# ===== إعدادات Hybrid =====
HYBRID = {
    'tfidf_weight': 0.3,
    'bm25_weight': 0.4,
    'embedding_weight': 0.3,
    'fusion_method': 'rrf',
    'rrf_k': 60,
}

# ===== إعدادات الاسترجاع =====
RETRIEVAL = {
    'top_k': 100,
    'top_k_display': 10
}

# ===== إعدادات Query Refinement =====
QUERY_REFINEMENT = {
    'use_synonyms': True,
    'use_spell_check': True,
    'use_history_weighting': True,
    'max_history': 5,
    'synonym_topn': 1,
}

# ===== إعدادات Multilingual =====
MULTILINGUAL = {
    'translation_model': 'Helsinki-NLP/opus-mt-ar-en',
    'detect_language': True,
}

# ===== الأسماء المعروضة =====
DATASET_NAMES = {
    'dataset2': 'Webis Touché 2020',
}

MODELS_LIST = ['tfidf', 'bm25', 'embedding', 'hybrid_serial', 'hybrid_parallel']

# ===== MongoDB Configuration =====
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "ir_project"
MONGO_COLLECTIONS = {
    "dataset2": "documents_dataset2",
}