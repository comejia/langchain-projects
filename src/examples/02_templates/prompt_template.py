from langchain_core.prompts import PromptTemplate

template = "Eres un experto en marketing. Sugiere un eslogan creativo para un producto {product}"

prompt = PromptTemplate(input_variables=["product"], template=template)

prompt_full = prompt.format(product="café orgánico")
print(prompt_full)
