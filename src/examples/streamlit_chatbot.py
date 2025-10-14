from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
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

# Sidebar de configuración
with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"])

    chat_model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=openai_api_key
    )

# Inicializar el historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Se crea el template con comportamiento especifico
prompt_template = PromptTemplate(
    input_variables=["message", "history"],
    template="""
    Eres un asistente útil y amigable llamado ChatBot Pro.
    Historial de conversación: {history}
    Responde de manera clara y concisa a la siguiente pregunta: {message}
    """
)

# Cadena usando LCEL (LangChain Expression Language)
chain = prompt_template | chat_model

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

    # Agregar el mensaje al historial
    #st.session_state.messages.append(HumanMessage(content=input))

    # Generar respuesta usando el llm
    #response = chat_model.invoke(st.session_state.messages)

    # Mostrar la respuesta en la interfaz
    #with st.chat_message("assistant"):
     #   st.markdown(response.content)

    #st.session_state.messages.append(AIMessage(content=response.content))
