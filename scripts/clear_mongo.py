import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from pymongo import MongoClient

client = MongoClient(config.MONGO_URI)
client[config.MONGO_DB_NAME].drop_collection(config.MONGO_COLLECTIONS["dataset2"])
print("✅ تم حذف الـ collection القديمة")
