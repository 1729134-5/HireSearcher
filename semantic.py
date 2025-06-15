from sentence_transformers import SentenceTransformer
import numpy as np

# Carrega o modelo local (isso faz download apenas na primeira vez)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts):
    """
    Gera embeddings localmente usando o modelo all-MiniLM-L6-v2.
    texts: lista de strings
    return: lista de vetores (floats)
    """
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()

def cosine_similarity(a, b):
    vec1 = np.array(a)
    vec2 = np.array(b)
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
