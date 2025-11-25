from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)


vector_db = Chroma(
    embedding_function=embedding,
    persist_directory="/home/comejia/projects/langchain-project/src/examples/rag/chroma_db",
)

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

query = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"

results = retriever.invoke(query)

print("Top 2 documentos más similares:")
for i, doc in enumerate(results, start=1):
    print(f"Contenido {i}: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
