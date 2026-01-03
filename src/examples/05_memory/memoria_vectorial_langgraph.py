from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from langchain_chroma import Chroma
import uuid
from langgraph.checkpoint.memory import MemorySaver

CHROMADB_PATH = (
    "/home/comejia/projects/langchain-project/src/examples/05_memory/chroma_db"
)


# Configuración básica del LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Configuración del vectorstore
vectorstore = Chroma(
    collection_name="memory",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory=CHROMADB_PATH,
)


client = chromadb.PersistentClient(path=CHROMADB_PATH)

collection = client.get_collection(name="memory")


def save_memory(text):
    """Guarda información relevante del usuario en la base de datos vectorial"""
    try:
        collection.add(
            documents=[text],
            metadatas=[{"source": "user_input"}],
            ids=[str(uuid.uuid4())],
        )

        print(f"✅ Guardado en memoria: {text}")
    except Exception as e:
        print(f"Error al guardar memoria: {e}")


def search_memory(query, k=3):
    """Recupera la información relevante del usuario en la base de datos vectorial"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=k,
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"Error al buscar memoria: {e}")
        return []


def chatbot_node(state):
    """Nodo principal del grafo."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # 1. Buscar memorias relevantes
    memories = search_memory(last_message)

    # 2. Crear prompt con memorias relevantes
    system_content = "Eres un asistente que recuerda información importante del usuario"
    if memories:
        system_content += "\n\nInformación que recuerdas:"
        for memory in memories:
            system_content += f"\n- {memory}"

    # 3. Generar respuesta
    messages_with_system_prompt = [SystemMessage(content=system_content)] + messages
    response = llm.invoke(messages_with_system_prompt)

    # 4. Guardar información relevante del usuario en la base de datos vectorial
    message_lower = last_message.lower()
    if "me llamo" in message_lower:
        save_memory(f"El usuario se llama: {last_message}")
    elif any(
        phrase in message_lower
        for phrase in [
            "trabajo en",
            "trabajo como",
            "soy programador",
            "soy estudiante",
        ]
    ):
        save_memory(f"Trabajo del usuario: {last_message}")
    elif "me gusta" in message_lower:
        save_memory(f"Le gusta: {last_message}")
    elif "vivo en" in message_lower or "soy de" in message_lower:
        save_memory(f"Ubicación del usuario: {last_message}")

    return {"messages": [response]}


workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)


def chat(message, thread_id="session_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    response = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return response["messages"][-1].content


def show_memories():
    """Función auxiliar para ver todas las memorias guardadas del usuario."""

    try:
        all_memories = collection.get()
        if all_memories:
            print("[+] Memorias guardadas:")
            for i, memory in enumerate(all_memories["documents"], 1):
                print(f"{i}. {memory}")
        else:
            print("[-] No hay memorias guardadas aun.")
    except Exception as e:
        print(f"Error al mostrar memorias: {e}")


if __name__ == "__main__":
    print("Chat en terminal")
    print("----------------")
    session_id = "session_terminal"

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("\nGoodbye!")
            break

        if user_input.lower() == "memorias":
            show_memories()
            continue

        response = chat(user_input, session_id)

        print("Assistant:", response)
        print("----------------")
