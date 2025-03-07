import chromadb

# Initialize the Chroma client
chroma_client = chromadb.PersistentClient(path="backend/chroma_db")

# List all collections
collection_names = chroma_client.list_collections()

# Print the collection names
print("Collections in Chroma DB:")
for name in collection_names:
    print(f"- {name}")