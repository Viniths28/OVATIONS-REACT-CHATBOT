import chromadb

# Initialize the Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Access the 'ovations_data' collection
collection_name = "langchain"
collection = chroma_client.get_collection(collection_name)

# Get the number of items in the collection
print(f"Number of items in collection '{collection_name}': {collection.count()}")

# Retrieve the first few items
results = collection.peek()
print("Sample items in the collection:")
for i, doc in enumerate(results["documents"]):
    print(f"Document {i + 1}:")
    print(f"ID: {results['ids'][i]}")
    print(f"Document: {doc}")
    print(f"Metadata: {results['metadatas'][i]}")
    print("-" * 50)