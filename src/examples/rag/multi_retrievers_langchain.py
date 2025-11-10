from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
llm = ChatOpenAI(model=openai_model, temperature=0, api_key=openai_api_key)


vector_db = Chroma(
    embedding_function=embedding,
    persist_directory="/home/comejia/projects/langchain-project/src/examples/rag/chroma_db",
)

base_retriever = vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 2}
)
multi_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

query = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"

results = multi_retriever.invoke(query)

print("Top k documentos más similares:")
for i, doc in enumerate(results, start=1):
    print(f"Contenido {i}: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
