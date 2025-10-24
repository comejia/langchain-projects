from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} especializado en {specialty}. Responde de manera {type}"
)

human_template = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {topic} es: {question}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])

# Test chat
messages = chat_prompt.format_messages(
    rol="nutricionista",
    specialty="dietas veganas",
    type="profesional pero accesible",
    topic="proteinas vegetales",
    question="¿Cuáles son las mejores fuentes de proteina vegana para un atleta profesional?"
)

for message in messages:
    print(f"{message.content}")