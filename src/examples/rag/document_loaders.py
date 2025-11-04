from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, GoogleDriveLoader


loader = PyPDFLoader(
    "/home/comejia/projects/langchain-project/src/examples/rag/Profile.pdf"
)

pages = loader.load()

for i, page in enumerate(pages):
    print(f"=== Pagina {i+1} ===")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")


loader = WebBaseLoader("https://techmind.ac/")

data = loader.load()

print(data)
