"""
Upload Raw Documents to MongoDB
يرفع الوثائق الخام (raw text) إلى MongoDB بحيث يمكن قراءتها وقت الـ query بسرعة عبر doc_id
يشتغل offline بعد التحضير، ومرة واحدة فقط (أو عند تغيير الداتا)
"""

import os
import json
import sys
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def load_documents(docs_path: str) -> dict:
    """يحمل ملف الوثائق ويرجع dict: {doc_id: text}"""
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                normalized[str(v.get("doc_id", k))] = str(v.get("text", v.get("body", "")))
            else:
                normalized[str(k)] = str(v)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                doc_id = str(item.get("doc_id") or item.get("id") or i)
                text = str(item.get("text", item.get("body", "")))
                normalized[doc_id] = text
            else:
                normalized[str(i)] = str(item)
    return normalized


def upload_to_mongo(dataset_key: str, docs_path: str, batch_size: int = 1000):
    client = MongoClient(config.MONGO_URI)
    db = client[config.MONGO_DB_NAME]
    collection = db[config.MONGO_COLLECTIONS[dataset_key]]

    # إنشاء index على doc_id لتسريع القراءة
    collection.create_index("doc_id", unique=True)

    print(f"[Mongo] تحميل الوثائق من {docs_path} ...")
    documents = load_documents(docs_path)
    print(f"[Mongo] عدد الوثائق: {len(documents)}")

    operations = []
    for doc_id, text in tqdm(documents.items(), desc=f"Uploading {dataset_key}"):
        operations.append(
            UpdateOne(
                {"doc_id": doc_id},
                {"$set": {"doc_id": doc_id, "text": text}},
                upsert=True,
            )
        )
        if len(operations) >= batch_size:
            collection.bulk_write(operations)
            operations = []

    if operations:
        collection.bulk_write(operations)

    print(f"[Mongo] ✅ تم رفع {len(documents)} وثيقة إلى collection '{collection.name}'")


if __name__ == "__main__":
    upload_to_mongo("dataset2", config.DATASET2_DOCS)
    print("\n🎉 انتهى الرفع بنجاح!")