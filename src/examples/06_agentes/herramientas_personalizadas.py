from langchain_core.tools import tool
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from operator import attrgetter
from typing import Tuple


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


@tool("user_db_tool", response_format="content_and_artifact")
def herramienta_personalizada(query: str) -> Tuple[str, int]:
    """Consulta la base de datos de usuarios de la empresa."""

    return f"Respuesta a tu consulta: {query}", 10


def herramienta_personalizada2(query: str) -> str:
    """Consulta la base de datos de usuarios de la empresa."""

    return f"Respuesta a tu consulta: {query}"


output = herramienta_personalizada.invoke("¿Cuántos usuarios hay en la empresa?")
print(output)
print(f"Nombre de la herramienta: {herramienta_personalizada.name}")
print(f"Descripción de la herramienta: {herramienta_personalizada.description}")


my_tool = StructuredTool.from_function(herramienta_personalizada2)
output = my_tool.invoke("Consulta personalizada")
print(output)


# Integrando el llm con la herramienta
llm_with_tools = llm.bind_tools([herramienta_personalizada])

response = llm_with_tools.invoke(
    "Genera un resumen de la información que hay en la base de datos"
)
print(response)


chain = llm_with_tools | attrgetter("tool_calls") | herramienta_personalizada.map()

response = chain.invoke(
    "Genera un resumen de la información que hay en la base de datos"
)
print(response)
