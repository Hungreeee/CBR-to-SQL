# %%
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# %%
model = BGEM3FlagModel("BAAI/bge-m3",  use_fp16=True) 

# %%
sentences = [
    "How many peas have I eaten today?",
    "How many peas have I had today?"
]

embeddings = [model.encode(sentence, return_dense=True)["dense_vecs"] for sentence in sentences]

# %%
def cosine_similarity(x, y):
    return np.dot(x, y) / np.linalg.norm(x) * np.linalg.norm(y)

cosine_similarity(embeddings[0], embeddings[1])

# %%

