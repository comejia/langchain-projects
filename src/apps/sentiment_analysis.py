from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
import json

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    return text.strip()[:500]


def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content


def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    
    Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}


def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"],
    }


# Configuración del modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

preprocessor = RunnableLambda(preprocess_text)
summarizer = RunnableLambda(generate_summary)
analyzer = RunnableLambda(analyze_sentiment)
merger = RunnableLambda(merge_results)
parallel_analysis = RunnableParallel(
    {"resumen": summarizer, "sentimiento_data": analyzer}
)

chain = preprocessor | parallel_analysis | merger


reviews = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde.",
]

# Process in parallel
results = chain.batch(reviews)

for review, result in zip(reviews, results):
    print(f"Texto: {review}")
    print(f"Resultado: {result}")
    print("-" * 50)
