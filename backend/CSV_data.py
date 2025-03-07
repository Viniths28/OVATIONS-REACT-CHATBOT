import os
import pandas as pd
import chromadb
from langchain_openai import OpenAIEmbeddings  
from langchain_openai import AzureOpenAIEmbeddings  
from langchain_chroma import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="chroma_db") 
openai_embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


#azure_embeddings = AzureOpenAIEmbeddings(
 #   azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  #  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
   # azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
#) 



# adding CSV Data

csv_folder = 'csv_files'
documents = []

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        loader = CSVLoader(os.path.join(csv_folder, file), encoding='utf-8')
        documents.extend(loader.load())

print(f"Total documents loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 100)
texts =text_splitter.split_documents(documents)

print(f"Total text chunks after splitting: {len(texts)}")

Vector_store = Chroma.from_documents(texts, openai_embeddings, client= chroma_client,persist_directory="chroma_db")
print("Data Loaded Successfully ")

