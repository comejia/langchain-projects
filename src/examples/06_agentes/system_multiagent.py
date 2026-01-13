from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Definir herramientas personalizadas
@tool
def search_web(query: str) -> str:
    """Buscar información en la web."""
    return f"Resultado de búsqueda para '{query}'"


@tool
def calculate(expression: str) -> str:
    """Realizar cálculos matemáticos."""
    return f"Resultado: {eval(expression)}"


# Crear agentes especializados
search_agent = create_react_agent(
    model=model,
    tools=[search_web],
    prompt="Eres un especialista solo en investigación web. No realices cálculos ni otras tareas que no sean de investigacion.",
    name="investigador",
)
calculate_agent = create_react_agent(
    model=model,
    tools=[calculate],
    prompt="Eres un especialista en cálculos matemáticos.",
    name="calculador",
)

# Crear supervisor que coordina a los agentes
supervisor_graph = create_supervisor(
    model=model,
    agents=[search_agent, calculate_agent],
    prompt="Eres un supervisor que delega tareas a los agentes especializados segun el tipo de consulta.",
)


supervisor = supervisor_graph.compile()

# Uso del sistema multi-agente
response = supervisor.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Busca informacion sobre pi y calcula su valor multiplicado por 2",
            }
        ]
    }
)

for message in response["messages"]:
    print(message.content)
