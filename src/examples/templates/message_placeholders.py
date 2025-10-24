from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


history = [
    HumanMessage(content="¿Cúal es la capital de Francia?"),
    AIMessage(content="La capital de Francia es París"),
    HumanMessage(content="¿Y cuantos habitantes tiene?"),
    AIMessage(content="Paris tiene aproximadamente 2.2 millones de habitantes en la ciudad"),
]

# Test chat
messages = chat_prompt.format_messages(
    history=history,
    question="¿Dime algo interesante de la arquitectura?"
)

for message in messages:
    print(f"{message.content}")

