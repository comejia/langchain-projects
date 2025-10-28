from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

llm = ChatOpenAI(model=openai_model, temperature=0.7, api_key=openai_api_key)

template = PromptTemplate(
    input_variables=["name"],
    template="Saluda al usuario con su nombre.\nUsuario: {name}\nAsistente",
)

# chain = LLMChain(llm=llm, prompt=template) # Deprecated
chain = template | llm  # New way to create a chain

response = chain.invoke({"name": "Juan"})

print("Respuesta: ", response.content)
