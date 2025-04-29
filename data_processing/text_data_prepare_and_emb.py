import os
import json
from sentence_transformers import SentenceTransformer
from config import Config

with open(f'../{Config.parsed_full_data_path}', 'r', encoding='utf-8') as f:
    articles = json.load(f)

output_dir = f'../{Config.text_dataset_path}'
os.makedirs(output_dir, exist_ok=True)


documents = []
metadatas = []
ids = []

seen_titles = set()

for article in articles:
    title = article.get('title')
    text = article.get('text')

    if (
        title is not None
        and text is not None
        and 'a message from' not in title.lower()
        and title not in seen_titles
    ):
        documents.append(text)
        metadatas.append({"url": article['url'], "issue_id": article['issue']})
        ids.append(title)
        seen_titles.add(title)


print(f"Number of entries: {len(documents)}")
print(f"Number of metadatas: {len(metadatas)}")
print(f"Number of ids: {len(ids)}")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

model = SentenceTransformer('all-MiniLM-L6-v2')#.to(device)

embeddings = []
batch_size = Config.embedding_batch_size

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    batch_embeddings = model.encode(batch, convert_to_tensor=True)#, device=device)
    embeddings.extend(batch_embeddings.cpu().numpy())

embeddings = [emb.tolist() for emb in embeddings]

export_data = {
    "documents": documents,
    "embeddings": embeddings,
    "metadatas": metadatas,
    "ids": ids
}

with open(f'../{Config.text_dataset_path}/text_embeddings.json', 'w') as f:
    json.dump(export_data, f)

print(f"Data exported to {Config.text_dataset_path}/text_embeddings.json")