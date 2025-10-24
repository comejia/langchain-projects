from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso."),
    ("human", "{text}")
])

# Test chat
messages = chat_prompt.format_messages(text="Hola, ¿cómo estás?")

for message in messages:
    print(f"{type(message)}: {message.content}")
