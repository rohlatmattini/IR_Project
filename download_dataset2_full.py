import os
import json
import ir_datasets
from tqdm import tqdm

def download_and_save(dataset_name, output_folder):
    print("="*50)
    print(f"🚀 البدء بتحميل وحفظ المجموعة: {dataset_name}")
    print(f"📍 المسار المستهدف: {output_folder}")
    print("="*50)

    os.makedirs(output_folder, exist_ok=True)
    dataset = ir_datasets.load(dataset_name)

    queries_path = os.path.join(output_folder, 'queries.json')
    if not os.path.exists(queries_path):
        queries_dict = {q.query_id: q.text for q in dataset.queries_iter()}
        with open(queries_path, 'w', encoding='utf-8') as f:
            json.dump(queries_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ تم حفظ {len(queries_dict)} استعلام.")
    else:
        print("ℹ️ ملف الاستعلامات موجود مسبقاً، تخطي...")

    qrels_path = os.path.join(output_folder, 'qrels.json')
    if not os.path.exists(qrels_path):
        qrels_list = [
            {"query_id": q.query_id, "doc_id": q.doc_id, "relevance": q.relevance}
            for q in dataset.qrels_iter()
        ]
        with open(qrels_path, 'w', encoding='utf-8') as f:
            json.dump(qrels_list, f, ensure_ascii=False, indent=4)
        print(f"✅ تم حفظ {len(qrels_list)} علاقة تقييم.")
    else:
        print("ℹ️ ملف الـ Qrels موجود مسبقاً، تخطي...")

    docs_path = os.path.join(output_folder, 'documents.json')
    if not os.path.exists(docs_path):
        try:
            total = dataset.docs_count()
        except Exception:
            total = None

        docs_dict = {}
        for doc in tqdm(dataset.docs_iter(), total=total):
            text_content = getattr(doc, 'text', getattr(doc, 'body', ''))
            docs_dict[doc.doc_id] = text_content

        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(docs_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ تم حفظ {len(docs_dict)} وثيقة بنجاح.")
    else:
        print("ℹ️ ملف الوثائق موجود مسبقاً، تخطي...")


if __name__ == "__main__":
    download_and_save('beir/webis-touche2020', './data/dataset2')
    print("\n🎉 انتهى التحميل بنجاح!")