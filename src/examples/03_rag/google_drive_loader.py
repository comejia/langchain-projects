from langchain_google_community import GoogleDriveLoader

credentials_path = (
    "/home/comejia/projects/langchain-project/src/examples/rag/credentials.json"
)
token_path = "/home/comejia/projects/langchain-project/src/examples/rag/token.json"

loader = GoogleDriveLoader(
    folder_id="1TOqon2698K2FOHY2dQE8XgyAxXDSDPMn",
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True,
)

documents = loader.load()

print(f"Metadata: {documents[0].metadata}")
print(f"Contenido: {documents[0].page_content}")
