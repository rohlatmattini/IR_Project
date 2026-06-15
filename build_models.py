import sys
sys.path.append(r"C:\Users\LOQ\Desktop\IR_Project")

from services.retrieval_service.bm25_model import get_bm25_model
from services.retrieval_service.tfidf_model import get_tfidf_model

print("Building BM25 dataset1...")
get_bm25_model("dataset1")

print("Building TF-IDF dataset1...")
get_tfidf_model("dataset1")

print("Building BM25 dataset2...")
get_bm25_model("dataset2")

print("Building TF-IDF dataset2...")
get_tfidf_model("dataset2")

print("Done!")