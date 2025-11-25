from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

text1 = "La capital de Francia es París."
text2 = "París es la capital de Francia."
text3 = "París es un nombre común para mascotas."

embedding1 = embeddings.embed_query(text1)
embedding2 = embeddings.embed_query(text2)
embedding3 = embeddings.embed_query(text3)

print("Dimensión del embedding 1:", len(embedding1))
print("Dimensión del embedding 2:", len(embedding2))

cos_similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)

cos_similarity_2 = np.dot(embedding1, embedding3) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding3)
)

print(f"Similitud entre text 1 y 2: {cos_similarity:.3f}")

print(f"Similitud entre text 1 y 3: {cos_similarity_2:.3f}")
