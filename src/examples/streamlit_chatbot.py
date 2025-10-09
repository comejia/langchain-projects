from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


# Configuración de la aplicación Streamlit
st.set_page_config(page_title="Chatbot básico", page_icon="🤖")
st.title("Chatbot básico con LangChain")
st.markdown("Este es un *chatbot* básico utilizando LangChain y Streamlit!")

llm = ChatOpenAI(
    model=openai_model,
    temperature=0.6,
    api_key=openai_api_key
)