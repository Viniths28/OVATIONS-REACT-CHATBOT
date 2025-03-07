import os
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Debugging: Print embeddings for a sample text
sample_text = "This is a sample text for debugging embeddings."
sample_embedding = embeddings.embed_query(sample_text)
print("Sample Embedding:", sample_embedding)


# Debugging: Print the first few documents in Chroma DB

# vector_store = Chroma.from_documents(texts, embeddings, client=chroma_client, persist_directory="chroma_db")

vector_store = Chroma(
    client=chroma_client,
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

collection = chroma_client.get_collection(name="langchain")  # Replace with your collection name
all_documents = collection.get(include=["documents", "embeddings"])
print("Total Documents in Chroma DB:", len(all_documents["documents"]))
print("First 3 Documents in Chroma DB:")
for i, doc in enumerate(all_documents["documents"][:3]):
    print(f"Document {i + 1}: {doc}")

# Create retriever from Chroma vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Fetch up to 10 chunks

# Debugging: Test the retriever with a sample query
sample_query = "What is Ovations Speakers Bureau? "
relevant_chunks = vector_store.similarity_search(sample_query, k=10)  # Fetch up to 10 chunks
print("Retrieved Chunks for Sample Query:")
for i, chunk in enumerate(relevant_chunks):
    print(f"Chunk {i + 1}: {chunk.page_content}")

# Debugging: Print query embedding and retrieved chunk embeddings
query_embedding = embeddings.embed_query(sample_query)
#print("Query Embedding:", query_embedding)

#for i, chunk in enumerate(relevant_chunks):
 #   chunk_embedding = embeddings.embed_query(chunk.page_content)
  #  print(f"Chunk {i + 1} Embedding:", chunk_embedding)

# Initialize the language model (LLM)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Debugging: Test the QA chain with a sample query
response = qa_chain.invoke({"query": sample_query})
print("QA Chain Response:", response["result"])
#print("Source Documents:")
#for doc in response["source_documents"]:
 #   print(doc.page_content)