from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


class TextAnalysis(BaseModel):
    summary: str = Field(description="Resumen breve del texto.")
    sentiment: str = Field(
        description="Sentimiento del texto (positivo, negativo o neutral)."
    )


llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0.6)

structured_llm = llm.with_structured_output(TextAnalysis)

test = "Me encanto la nueva pelicula de accion, tiene muchos efectos especiales."

result = structured_llm.invoke(f"Analiza el siguiente texto: {test}")

print(f"Tipo: {type(result)}")
print(f"Contenido: {result.model_dump_json()}")
