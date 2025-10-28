from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

llm = ChatOpenAI(model=openai_model, temperature=0.7, api_key=openai_api_key)

ask = "¿En que año llego el ser humano a la luna por primera vez?"
print(ask)

response = llm.invoke(ask)
print("Respuesta: ", response.content)
