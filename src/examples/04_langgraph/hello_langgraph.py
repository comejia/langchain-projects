from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Definir el esquema del estado
class State(TypedDict):
    original_text: str
    mayus_text: str
    lenght: int


# 2. Crear el grafo de estado
graph = StateGraph(State)


# 3. Definir las funciones de los nodos
def to_mayus(state: State):
    text = state["original_text"]
    return {"mayus_text": text.upper()}


def count_chars(state: State):
    text = state["mayus_text"]
    return {"lenght": len(text)}


# 4. Añadir los nodos al grafo
graph.add_node("Mayus", to_mayus)
graph.add_node("Counter", count_chars)

# 5. Conectar los nodos en secuencia
graph.add_edge(START, "Mayus")
graph.add_edge("Mayus", "Counter")
graph.add_edge("Counter", END)

# 6. Compilar el grafo
compiled_graph = graph.compile()

# 7. Invocar el grafo con un estado inicial
initial_state = {"original_text": "Hello, world!"}
result = compiled_graph.invoke(initial_state)

# 8. Imprimir el resultado
print(result)
