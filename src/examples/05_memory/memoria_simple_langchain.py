from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

store = {}


def get_sesion_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


session_id = "session_terminal"

# Cadena con memoria automatica por sesion
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_sesion_history,
    input_messages_key="input",
    history_messages_key="history",
)

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

    response = chain_with_history.invoke(
        input={"input": user_input}, config={"configurable": {"session_id": session_id}}
    )

    print("Assistant:", response.content)
