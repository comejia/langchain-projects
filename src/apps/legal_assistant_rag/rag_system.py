from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import settings
from prompts import MULTI_QUERY_PROMPT, RAG_TEMPLATE
import streamlit as st


@st.cache_resource
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

    # Retriever base MMR
    base_retriever = vector_db.as_retriever(
        search_type=settings.search_type,
        search_kwargs={
            "k": settings.search_k,
            "lambda_mult": settings.mmr_diversity_lambda,
            "fetch_k": settings.mmr_fetch_k,
        },
    )

    # Retriever adicional
    similarity_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.search_k,
        },
    )

    # MultiQueryRetriever con prompt personalizado
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm_queries, prompt=multi_query_prompt
    )

    # Ensemble Retriever si está habilitado
    if settings.enable_hybrid_search:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3],
            # similarity_threshold=settings.similarity_threshold,
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

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
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, final_retriever


def query_rag(question):
    try:
        rag_chain, retriever = init_rag_system()

        # Realizar la consulta
        response = rag_chain.invoke(question)

        # Obtener los documentos recuperados
        docs = retriever.invoke(question)

        docs_info = []
        for i, doc in enumerate(docs[: settings.search_k], start=1):
            doc_info = {
                "fragment": i,
                "content": doc.page_content[:1000]
                if len(doc.page_content) > 1000
                else doc.page_content,
                "source": doc.metadata.get("source", "No especificado").split("/")[-1],
                "page": doc.metadata.get("page", "No especificado"),
            }
            docs_info.append(doc_info)

        return response, docs_info

    except Exception as e:
        error_msg = f"Error al procesar la consulta: {str(e)}"
        return error_msg, []


def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": settings.search_type.upper(),
        "documentos": settings.search_k,
        "diversidad": settings.mmr_diversity_lambda,
        "candidatos": settings.mmr_fetch_k,
        "umbral": None,
    }
