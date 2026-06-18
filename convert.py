import json

with open('./data/dataset2/documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs_list = [{"doc_id": doc_id, "text": text} for doc_id, text in data.items()]

with open('./data/dataset2/documents_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(docs_list, f, ensure_ascii=False, indent=4)

print("تم التحويل بنجاح")