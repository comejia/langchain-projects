from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

workflow = StateGraph(
    state_schema=MessagesState,
)


def chatbot_node(state: MessagesState):
    """Nodo que procesa mensajes y genera respuestas."""
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas"
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

conn = sqlite3.connect("history.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(checkpointer=memory)


def chat(message, thread_id="session_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    response = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return response["messages"][-1].content


session_id = "session_terminal"

if __name__ == "__main__":
    print("Chat en terminal")
    print("----------------")

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

        response = chat(user_input, session_id)

        print("Assistant:", response)
