from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

loader = PyPDFDirectoryLoader(
    "/home/comejia/projects/langchain-project/src/examples/rag/contratos"
)

documents = loader.load()

print(f"Número de documentos cargados: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = text_splitter.split_documents(documents)

print(f"Número de fragmentos después del split: {len(docs)}")

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="/home/comejia/projects/langchain-project/src/examples/rag/chroma_db",
)

query = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"

results = vector_db.similarity_search(query, k=2)

print("Top 2 documentos más similares:")
for i, doc in enumerate(results, start=1):
    print(f"Contenido {i}: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
