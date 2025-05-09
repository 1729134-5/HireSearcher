import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # Carrega HUGGINGFACE_TOKEN do .env
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Endpoint da API de Inferência para o modelo all-MiniLM-L6-v2
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings(texts):
    """
    Retorna embeddings para uma lista de textos usando a API de Inferência do Hugging Face.
    texts: list de strings
    Retorna: lista de vetores (list of floats) correspondente a cada texto.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    response = requests.post(HF_EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    embeddings = response.json()
    return embeddings  # lista de listas de floats

def cosine_similarity(a, b):
    """
    Calcula similaridade de cosseno entre dois vetores a e b.
    """
    vec1 = np.array(a)
    vec2 = np.array(b)
    # Evita divisão por zero
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(sim)
