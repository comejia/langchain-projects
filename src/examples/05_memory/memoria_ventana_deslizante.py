from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


class WindowedState(MessagesState):
    pass


workflow = StateGraph(state_schema=WindowedState)

trimmer = trim_messages(
    strategy="last",
    max_tokens=4,
    token_counter=len,
    start_on="human",
    include_system=True,
)


def chatbot_node(state: MessagesState):
    """Nodo que procesa mensajes y genera respuestas."""
    trim_messages = trimmer.invoke(state["messages"])
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas"
    messages = [SystemMessage(content=system_prompt)] + trim_messages
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

memory = MemorySaver()

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
