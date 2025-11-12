from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = Field(alias="OPENAI_API_KEY")
    query_model: str
    generation_model: str
    embedding_model: str

    # Vector DB path
    db_path: str = "/home/comejia/projects/langchain-project/src/apps/legal_assistant_rag/chroma_db"

    # Retriever settings
    search_type: str = "mmr"
    mmr_diversity_lambda: float = 0.7
    mmr_fetch_k: int = 20
    search_k: int = 2


settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump())
