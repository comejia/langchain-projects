from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Annotated
from src.config.settings import settings
from operator import add


llm = ChatOpenAI(model=settings.query_model, api_key=settings.api_key, temperature=0.3)


# Definición del Estado
class State(TypedDict):
    notes: str  # Texto original de entrada
    participants: List[str]  # Participantes identificados
    topics: List[str]  # Temas principales
    action_items: List[str]  # Acciones y responsables
    minutes: str  # Minuta formal
    summary: str  # Resumen ejecutivo
    logs: Annotated[List[str], add]


# Nodos del workflow
def extract_participants(state: State) -> State:
    prompt = f"""
    Analiza las siguientes notas de reunión y extrae únicamente los nombres de los participantes.
    
    Notas: {state["notes"]}
    
    Instrucciones:
    - Responde SOLO con nombres separados por comas
    - No incluyas explicaciones adicionales
    - Formato: Juan García, María López, Carlos Ruiz
    """

    response = llm.invoke(prompt)
    participants = [p.strip() for p in response.content.split(",") if p.strip()]

    return {"participants": participants, "logs": ["Paso 1 completado"]}


def identify_topics(state: State) -> State:
    prompt = f"""
    Identifica los 3-5 temas principales discutidos en esta reunión.
    
    Notas: {state["notes"]}
    
    Instrucciones:
    - Responde SOLO con los temas principales
    - Evita categorias demasiado generales o específicas
    - Formato: Arquitectura del sistema; Plazos de entrega; Asignación de tareas
    """

    response = llm.invoke(prompt)
    topics = [t.strip() for t in response.content.split(";") if t.strip()]

    return {"topics": topics, "logs": ["Paso 2 completado"]}


def extract_actions(state: State) -> State:
    prompt = f"""
    Extrae las acciones específicas acordadas en la reunión, incluyendo el responsable si se menciona

    Notas: {state["notes"]}

    Instrucciones:
    - Responde SOLO con los acciones
    - Localizar compromisos y asignaciones de responsabilidad
    - Formato: Maria se encargará del backend | Carlos preparará la presentación | Próxima reunión el domingo

    Si no hay acciones claras, responde con "No se identificarón acciones específicas."
    """

    response = llm.invoke(prompt)

    if "No se identificarón" in response.content:
        return {"action_items": []}
    else:
        action_items = [a.strip() for a in response.content.split("|") if a.strip()]

    return {"action_items": action_items, "logs": ["Paso 3 completado"]}


def generate_minutes(state: State) -> State:
    participants = ", ".join(state["participants"])
    topics = "\n* ".join(state["topics"])
    actions = (
        "\n* ".join(state["action_items"])
        if state["action_items"]
        else "No se identificarón acciones específicas."
    )

    prompt = f"""
    Genera una minita formal y profesional basándote en la siguiente información:

    PARTICIPANTES: {participants}

    TEMAS DISCUTIDOS:
    * {topics}

    ACCIONES ACORDADAS:
    * {actions}

    NOTAS ORIGINALES: {state["notes"]}

    Instrucciones:
    - Responde con un máximo de 150 palabras en tono profesional
    1. Encabezado con tipo de reunión
    2. Lista de asistentes
    3. Puntos principales discutidos
    4. Acuerdos y próximos pasos

    Usa un tono formal y estructura clara.
    """

    response = llm.invoke(prompt)

    return {"minutes": response.content}


def create_summary(state: State) -> State:
    prompt = f"""
    Crea un resumen ejecutivo que capture la esencia de la reunión.

    Participantes: {", ".join(state["participants"])}
    Tema principal: {state["topics"][0] if state["topics"] else "General"}
    Acciones clave: {len(state["action_items"])} acciones definidas
    Minuta: {state["minutes"]}

    Instrucciones:
    - Responde con un máximo de 30 palabras
    - El resumen debe ser conciso y directo al punto.
    """

    response = llm.invoke(prompt)

    return {"summary": response.content}


# Contrucción del grafo
def create_workflow():
    workflow = StateGraph(State)

    workflow.add_node("extract_participants", extract_participants)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("extract_actions", extract_actions)
    workflow.add_node("generate_minutes", generate_minutes)
    workflow.add_node("create_summary", create_summary)

    workflow.add_edge(START, "extract_participants")
    workflow.add_edge("extract_participants", "identify_topics")
    workflow.add_edge("identify_topics", "extract_actions")
    workflow.add_edge("extract_actions", "generate_minutes")
    workflow.add_edge("generate_minutes", "create_summary")
    workflow.add_edge("create_summary", END)

    return workflow.compile()


if __name__ == "__main__":
    app = create_workflow()

    test_state = {
        "notes": "Reunión con Juan García y María López sobre el proyecto...",
        "participants": [],
        "topics": [],
        "action_items": [],
        "minutes": "",
        "summary": "",
    }

    result = app.invoke(test_state)
    print(result)
