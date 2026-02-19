from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector DB
client = PersistentClient(path="rag/chroma_db")
collection = client.get_collection(name="doc")

print("\nAI Document Question Answering (Evidence-Based)")
print("Type 'exit' to quit.\n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break

    # Embed question
    q_emb = model.encode(q).tolist()

    # Retrieve most relevant chunk
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=1
    )

    top_chunk = results["documents"][0][0]

    print("\nAI Answer (from document):")
    print(top_chunk)
