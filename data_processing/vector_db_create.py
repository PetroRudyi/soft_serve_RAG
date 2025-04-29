import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
import json
from config import Config

path = f"../{Config.vector_db_path}"
client = chromadb.PersistentClient(path=path)

# Text Client creation
def load_text_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    documents = data['documents']
    embeddings = data['embeddings']
    metadatas = data['metadatas']
    ids = data['ids']

    print(f"Loaded data from {file_path}")
    print(f"Number of entries: {len(documents)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    return documents, embeddings, metadatas, ids

docs, embs, metas, ids = load_text_data(f'../{Config.text_dataset_path}/text_embeddings.json')

text_collection = client.get_or_create_collection(name="text_collection")

def batch_add_to_collection(collection, documents, embeddings, metadatas, ids, batch_size=Config.batch_add_to_collection_batch_size):
    for i in range(0, len(documents), batch_size):
        # Slice the data into batches
        doc_batch = documents[i:i + batch_size]
        emb_batch = embeddings[i:i + batch_size]
        meta_batch = metadatas[i:i + batch_size]
        id_batch = ids[i:i + batch_size]

        # Add the batch to the collection
        collection.add(
            documents=doc_batch,
            embeddings=emb_batch,
            metadatas=meta_batch,
            ids=id_batch
        )
        print(f"Batch {i // batch_size + 1} added to the collection successfully.")

batch_add_to_collection(text_collection, docs, embs, metas, ids)

# Image processing
print('Start Image Preprocessing')
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

# create image collection
image_collection = client.get_or_create_collection(name="image_collection",
                                                   embedding_function = CLIP,
                                                   data_loader = image_loader)

ids = []
uris = []

dataset_folder=f"../{Config.image_dataset_path}"

for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    file_path = os.path.join(dataset_folder, filename)
    ids.append(str(i))
    uris.append(file_path)

image_collection.add(
    ids=ids,
    uris=uris
)

print("Images added to the database.")
