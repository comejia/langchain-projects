from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

tool = Tool(
    name="Python REPL",
    func=python_repl.run,
    description="Ejecuta codigo python en un interprete para calculos o logica matematica",
)

output = tool.invoke("print(2+2)")
print(output)
