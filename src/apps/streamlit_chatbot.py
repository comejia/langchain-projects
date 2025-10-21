from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


# Configuración de la aplicación Streamlit
st.set_page_config(page_title="Chatbot básico", page_icon="🤖")
st.title("🤖 Chatbot básico con LangChain")
st.markdown("Este es un *chatbot* utilizando LangChain y Streamlit!")

# Sidebar de configuración
with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"])

    personality = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal", 
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido"
        ]
    )

    # Recrear modelo
    chat_model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=openai_api_key
    )

    # Template dinámico basado en personalidad
    system_messages = {
        "Útil y amigable": "Eres un asistente útil y amigable llamado ChatBot Pro. Responde de manera clara y concisa.",
        "Profesional y formal": "Eres un asistente profesional y formal. Proporciona respuestas precisas y bien estructuradas.",
        "Casual y relajado": "Eres un asistente casual y relajado. Habla de forma natural y amigable, como un buen amigo.",
        "Experto técnico": "Eres un asistente experto técnico. Proporciona respuestas detalladas con precisión técnica.",
        "Creativo y divertido": "Eres un asistente creativo y divertido. Usa analogías, ejemplos creativos y mantén un tono alegre."
    }

    # ChatPromptTemplate con personalidad dinámica
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_messages[personality]),
        ("human", "Historial de conversación:\n{history}\n\nPregunta actual: {message}")
    ])

    # Cadena usando LCEL (LangChain Expression Language)
    chain = chat_prompt_template | chat_model


# Inicializar el historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    
    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)

# Nuevo chat
if st.button("📝 Nuevo chat"):
    st.session_state.messages = []
    st.rerun()

# Input de usuario
input = st.chat_input("Escribe tu mensaje: ")

if input:
    # Mostrar el mensaje mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(input)

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
 
            # Streaming de la respuesta
            for chunk in chain.stream({"message": input, "history": st.session_state.messages}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")  # El cursor parpadeante

            response_placeholder.markdown(full_response)

        # Se almacenan los mensajes
        st.session_state.messages.append(HumanMessage(content=input))
        st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")

