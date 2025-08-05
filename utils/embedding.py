from sentence_transformers import SentenceTransformer
from typing import List

def embed(
    texts: List[str], 
    model: str, 
    path: str, 
    is_query=False,
):
    embedder = SentenceTransformer(path if path else model)
    embeddings = embedder.encode(texts)
    return embeddings