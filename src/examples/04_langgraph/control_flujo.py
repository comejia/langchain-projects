from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# from src.config.settings import settings


# Definición del estado
class State(TypedDict):
    number: int
    result: str


graph = StateGraph(State)


# Definición de los nodos del workflow
def case_par(state: State) -> State:
    return {"result": "El número es par"}


def case_impar(state: State) -> State:
    return {"result": "El número es impar"}


graph.add_node("case_par", case_par)
graph.add_node("case_impar", case_impar)


# Definición de la función de control de flujo (routing)
def decision_branch(state: State) -> State:
    if state["number"] % 2 == 0:
        return "case_par"
    else:
        return "case_impar"


# Añadir el edge condicional al workflow
graph.add_conditional_edges(START, decision_branch)

# Conectar ambos casos al final
graph.add_edge("case_par", END)
graph.add_edge("case_impar", END)

compiled_graph = graph.compile()

# Probar el grafo con ejemplos
initial_state = {"number": 4}
result = compiled_graph.invoke(initial_state)
print(result["result"])
