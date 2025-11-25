from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Cargar el documento PDF
loader = PyPDFLoader(
    "/home/comejia/projects/langchain-project/src/examples/rag/quijote.pdf"
)
pages = loader.load()

# 2. Dividir el texto en chunks mas pequeños
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(pages)

# 3. Pasar el texto al LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
summaries = []

for chunk in chunks:
    response = llm.invoke(
        f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}"
    )
    summaries.append(response.content)

print("Resúmenes de los chunks:\n", summaries)

final_summary = llm.invoke(
    f"Resume los siguientes puntos importantes en un solo resumen coherente: {' '.join(summaries)}"
)

print(final_summary.content)
