import chromadb

# Initialize the Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Delete the 'langchain' collection
collection_name = "langchain"
try:
    chroma_client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
except Exception as e:
    print(f"Error deleting collection '{collection_name}': {e}")

# List all collections to verify deletion
collection_names = chroma_client.list_collections()
print("Collections in Chroma DB after deletion:")
for name in collection_names:
    print(f"- {name}")