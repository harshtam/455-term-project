import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity

# Load model
df = pd.read_csv("training_data.csv")
model = SentenceTransformer('nlp-recipe-finder')

# Encode recipes for retrieval
recipe_embeddings = model.encode(df['tags'].tolist(), convert_to_tensor=True)

# Sample query
query = "potato mushroom stew dinner easy-to-make"
query_embedding = model.encode(query, convert_to_tensor=True)

# Similarity ranking
sims = cosine_similarity(query_embedding, recipe_embeddings)
top_k = torch.topk(sims, k=10)
top_indices = top_k.indices.tolist()

# Return top results sorted by rating
results = df.iloc[top_indices]
results = results.sort_values(by='rating', ascending=False)
print(results[['recipe_id', 'name', 'tags', 'rating']])