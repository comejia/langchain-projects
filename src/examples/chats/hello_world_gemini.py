from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GEMINI_API_KEY")
google_model = os.getenv("GOOGLE_MODEL")

llm = ChatGoogleGenerativeAI(
        model=google_model,
        temperature=0.7,
        api_key=google_api_key
    )

ask = "¿En que año llego el ser humano a la luna por primera vez?"
print(ask)

response = llm.invoke(ask)
print("Respuesta: ", response.content)