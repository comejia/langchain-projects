from langchain_openai import ChatOpenAI
from models.cv_model import CVAnalyzer
from prompts.cv_prompts import create_system_prompts
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

model_config = {
    "model": openai_model,
    "api_key": openai_api_key,
    "temperature": 0.2,
}

def create_cv_evaluator():
    model = ChatOpenAI(**model_config)

    model_structured = model.with_structured_output(CVAnalyzer)
    chat_prompt = create_system_prompts()
    chain = chat_prompt | model_structured

    return chain

def evaluate_candidate(cv: str, job_description: str) -> CVAnalyzer:
    try:
        chain = create_cv_evaluator()

        result = chain.invoke({"texto_cv": cv, "descripcion_puesto": job_description})
        return result
    except Exception as e:
        return CVAnalyzer(
            name="Error en procesamiento.",
            experience_year=0,
            skills=[],
            education="",
            experience_key="",
            strengths=[],
            weaknesses=[],
            adjustment_percentage=0
        )
