from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

history = []

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

    response = chain.invoke({"input": user_input, "history": history})

    print("Assistant:", response.content)

    # Actualiza el historial de mensajes
    history.extend(
        [HumanMessage(content=user_input), AIMessage(content=response.content)]
    )
