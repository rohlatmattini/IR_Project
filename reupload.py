from pymongo import MongoClient
import json
import config

client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB_NAME]
collection = db[config.MONGO_COLLECTIONS['dataset2']]

collection.delete_many({})

with open('./data/dataset2/documents_fixed.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)

collection.insert_many(docs)
print(f"تم رفع {len(docs)} وثيقة")