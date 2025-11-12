from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from prompts import MULTI_QUERY_PROMPT, RAG_TEMPLATE


def init_rag_system():
    # Models
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.api_key
    )
    llm_queries = ChatOpenAI(
        model=settings.query_model, api_key=settings.api_key, temperature=0
    )
    llm_generation = ChatOpenAI(
        model=settings.generation_model, api_key=settings.api_key, temperature=0
    )

    # Vector DB
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=settings.db_path,
    )

    # Retriever MMR
    base_retriever = vector_db.as_retriever(
        search_type=settings.search_type,
        search_kwargs={
            "k": settings.search_k,
            "lambda_mult": settings.mmr_diversity_lambda,
            "fetch_k": settings.mmr_fetch_k,
        },
    )

    # MultiQueryRetriever con prompt personalizado
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm_queries, prompt=multi_query_prompt
    )

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, start=1):
            header = f"[FRAGMENTO {i}]"
            if doc.metadata:
                if "source" in doc.metadata:
                    source = (
                        doc.metadata["source"].split("/")[-1]
                        if "/" in doc.metadata["source"]
                        else doc.metadata["source"]
                    )
                    header += f" - Fuente: {source}"
                if "page" in doc.metadata:
                    header += f" - Página: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}\n")

        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": mmr_multi_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever
