import os

# ===== المسارات =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATASET1_DIR = os.path.join(DATA_DIR, 'dataset1')
DATASET2_DIR = os.path.join(DATA_DIR, 'dataset2')

DATASET1_DOCS   = os.path.join(DATASET1_DIR, 'documents.json')
DATASET1_QUERIES = os.path.join(DATASET1_DIR, 'queries.json')
DATASET1_QRELS  = os.path.join(DATASET1_DIR, 'qrels.json')

DATASET2_DOCS   = os.path.join(DATASET2_DIR, 'documents.json')
DATASET2_QUERIES = os.path.join(DATASET2_DIR, 'queries.json')
DATASET2_QRELS  = os.path.join(DATASET2_DIR, 'qrels.json')

# مجلدات الفهارس
INDEX_DIR = os.path.join(BASE_DIR, 'indexes')
INDEX1_DIR = os.path.join(INDEX_DIR, 'dataset1')
INDEX2_DIR = os.path.join(INDEX_DIR, 'dataset2')

# مجلدات حفظ النماذج
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(INDEX1_DIR, exist_ok=True)
os.makedirs(INDEX2_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== إعدادات المعالجة المسبقة =====
PREPROCESSING = {
    'remove_stopwords': True,
    'stemming': False,        # Stemming أو Lemmatization — مو الاثنين سوا
    'lemmatization': True,
    'lowercase': True,
    'remove_punctuation': True,
    'min_token_length': 2,
}

# ===== إعدادات BM25 =====
BM25_PARAMS = {
    'k1': 1.5,   # يتحكم بتشبع تكرار المصطلح
    'b': 0.75,   # يتحكم بتطبيع طول الوثيقة
}

# ===== إعدادات Embedding =====
EMBEDDING = {
    'model_name': 'all-MiniLM-L6-v2',   # نموذج sentence-transformers خفيف وسريع
    'batch_size': 64,
    'max_seq_length': 128,
}

# ===== إعدادات Hybrid =====
HYBRID = {
    # أوزان الدمج في الـ Parallel Hybrid
    'tfidf_weight': 0.3,
    'bm25_weight': 0.4,
    'embedding_weight': 0.3,

    # طريقة الدمج: 'linear'  أو  'rrf' (Reciprocal Rank Fusion)
    'fusion_method': 'rrf',
    'rrf_k': 60,  # ثابت RRF الافتراضي
}

# ===== إعدادات الاسترجاع =====
RETRIEVAL = {
    'top_k': 100,       # عدد النتائج المسترجعة لكل استعلام
    'top_k_display': 10 # عدد النتائج المعروضة في الواجهة
}

# ===== إعدادات Query Refinement =====
QUERY_REFINEMENT = {
    'use_synonyms': True,
    'use_spell_check': True,
    'use_history_weighting': True,
    'max_history': 5,           # عدد الاستعلامات السابقة المحفوظة
    'synonym_topn': 3,          # عدد المرادفات لكل كلمة
}

# ===== إعدادات Multilingual =====
MULTILINGUAL = {
    'translation_model': 'Helsinki-NLP/opus-mt-ar-en',  # عربي → إنجليزي
    'detect_language': True,
}

# ===== الأسماء المعروضة =====
DATASET_NAMES = {
    'dataset1': 'MS MARCO Passage',
    'dataset2': 'Webis Touché 2020',
}

MODELS_LIST = ['tfidf', 'bm25', 'embedding', 'hybrid_serial', 'hybrid_parallel']



# ============================================
# MongoDB Configuration (للـ Raw Documents فقط)
# ============================================
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "ir_project"

MONGO_COLLECTIONS = {
    "dataset1": "documents_dataset1",
    "dataset2": "documents_dataset2",
}