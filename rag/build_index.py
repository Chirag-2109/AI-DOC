from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("rag/sample_doc.txt") as f:
    text = f.read()

chunks = [c.strip() for c in text.split(".") if c.strip()]

client = PersistentClient(path="rag/chroma_db")

collection = client.get_or_create_collection(name="doc")

for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[emb],
        ids=[str(i)]
    )

print("Vector index built and saved.")
