from typing import TypedDict, Optional, List, Annotated
from operator import add
from langchain_openai import ChatOpenAI
from rag_system import VectorRAGSystem
from config import CHROMADB_PATH, SQLITEDB_PATH
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# Definición del Estado
class HelpdeskState(TypedDict):
    consulta: str
    categoria: str  # automatico o escalado
    respuesta_rag: Optional[str]
    confianza: float
    fuentes: List[str]
    contexto_rag: Optional[str]
    requiere_humano: bool
    respuesta_humano: Optional[str]
    respuesta_final: Optional[str]
    historial: Annotated[List[str], add]


class HelpdeskGraph:
    """Grafo del sistema Helpdesk"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH)
        self.graph = None

    def process_rag(self, state: HelpdeskState) -> HelpdeskState:
        """Busca el contexto de la consulta utilizando el sistema RAG."""

        query = state["consulta"]
        result = self.rag.buscar(query)

        return {
            "respuesta_rag": result["respuesta"],
            "confianza": result["confianza"],
            "fuentes": result["fuentes"],
            "contexto_rag": result["respuesta"],
            "historial": [
                "RAG ejecutado con MultiQueryRetriever",
                f"Confianza: {result['confianza']}",
                f"Fuentes consultadas: {len(result['fuentes'])}",
            ],
        }

    def classify_with_context(self, state: HelpdeskState) -> HelpdeskState:
        """Clasifica la consulta para responser automaticamente o escalar con el contexto del RAG."""

        query = state["consulta"]
        context_rag = state.get("contexto_rag", "")
        confidence = state.get("confianza", 0.0)

        prompt = ChatPromptTemplate.from_template(
            """Analiza esta consulta de helpdesk y decide si puede responderse automáticamente o necesita escalado:

CONSULTA DEL USUARIO: {query}

INFORMACIÓN ENCONTRADA EN LA BASE DE CONOCIMIENTO:
{context_rag}

CONFIANZA DE LA BÚSQUEDA: {confidence}

Criterios de decisión:
- AUTOMATICO: Si la información de la BD responde completamente la consulta, 
  tiene buena confianza (>0.6), y es un tema estándar/procedimiento conocido
  
- ESCALADO: Si la información es insuficiente, confianza baja, problema complejo/único,
  requiere acceso a sistemas internos, o involucra decisiones de negocio

Responde solo con "automatico" o "escalado" y una breve justificación (máximo 20 palabras):"""
        )

        try:
            response = self.llm.invoke(
                prompt.format(
                    query=query, context_rag=context_rag, confidence=confidence
                )
            )
            content = response.content.strip().lower()

            if "automatico" in content or "automático" in content:
                category = "automatico"
            elif "escalado" in content:
                category = "escalado"
            else:
                category = "automatico" if confidence >= 0.6 else "escalado"

            return {
                "categoria": category,
                "historial": [
                    f"Clasificación con contexto: {category}",
                    f"Justificación: {response.content}",
                ],
            }
        except Exception:
            category = "automatico" if confidence >= 0.6 else "escalado"
            return {
                "categoria": category,
                "historial": [
                    f"Error en la clasificación, usando confianza: {confidence}"
                ],
            }

    def preparate_scalation(self, state: HelpdeskState) -> HelpdeskState:
        """Prepara el escalado a un humano."""

        return {
            "requiere_humano": True,
            "historial": ["Escalado a agente humano - Esperando intervención"],
        }

    def process_response_human(self, state: HelpdeskState) -> HelpdeskState:
        """Procesa la respuesta del humano."""
        response_human = state.get("respuesta_humano", "")

        if response_human:
            return {
                "respuesta_final": response_human,
                "historial": ["Agente humano proporcionó respuesta."],
            }

        return {"historial": ["Esperando respuesta del agente humano."]}

    def generate_final_response(self, state: HelpdeskState) -> HelpdeskState:
        """Genera la respuesta final del sistema al ticket del usuario."""

        if state.get("respuesta_final"):
            return {"historial": ["Respuesta final proporcionada por agente humano."]}

        # Si no hay respuesta final, se genera con IA (usamos la respuesta del sistema RAG)
        response_rag = state.get("respuesta_rag") or ""
        sources = state.get("fuentes", [])

        response_final = response_rag
        if sources:
            sources_text = ", ".join(sources)
            response_final += f"\n\nFuentes consultadas: {sources_text}"

        return {
            "respuesta_final": response_final,
            "historial": ["Respuesta final generada automaticamente."],
        }

    # Funciones de enrutamiento + Human in the loop
    def decide_from_clasification(self, state: HelpdeskState) -> HelpdeskState:
        """Decide hacian donde ir depues de la clasificación con contexto RAG."""
        category = state.get("categoria", "escalado")

        if category == "automatico":
            return "respuesta_final"
        else:
            return "escalado"

    def decide_from_human(self, state: HelpdeskState) -> HelpdeskState:
        """Decide si continuar o esperar respuesta del humano."""
        response_human = state.get("respuesta_humano", "")

        if response_human:
            return "procesar_humano"
        else:
            return "esperar"

    def create_graph(self):
        """Crea el grafo del sistema con los nodos y control de flujo."""
        self.graph = StateGraph(HelpdeskState)

        # Añadir los nodos al grafo
        self.graph.add_node("rag", self.process_rag)
        self.graph.add_node("clasificar", self.classify_with_context)
        self.graph.add_node("escalado", self.preparate_scalation)
        self.graph.add_node("respuesta_final", self.generate_final_response)
        self.graph.add_node("procesar_humano", self.process_response_human)

        # Definir la estructura del grafo
        self.graph.add_edge(START, "rag")
        self.graph.add_edge("rag", "clasificar")

        # Edges condicionales del grafo
        self.graph.add_conditional_edges(
            "clasificar",
            self.decide_from_clasification,
            {"respuesta_final": "respuesta_final", "escalado": "escalado"},
        )

        self.graph.add_conditional_edges(
            "escalado",
            self.decide_from_human,
            {
                "procesar_humano": "procesar_humano",
                "esperar": END,  # Pausar la ejecución del grafo hasta que responda el humano
            },
        )

        self.graph.add_edge("procesar_humano", END)
        self.graph.add_edge("respuesta_final", END)

        return self.graph

    def compile_graph(self):
        """Compila el grafo con checkpointer."""
        if not self.graph:
            self.create_graph()

        conn = sqlite3.connect(f"{SQLITEDB_PATH}/helpdesk.db", check_same_thread=False)

        checkpointer = SqliteSaver(conn)

        compiled_graph = self.graph.compile(
            checkpointer=checkpointer, interrupt_before=["procesar_humano"]
        )

        return compiled_graph


def create_helpdesk():
    helpdesk = HelpdeskGraph()
    return helpdesk.compile_graph()
