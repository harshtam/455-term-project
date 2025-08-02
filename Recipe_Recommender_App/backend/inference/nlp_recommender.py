import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity

# Load model
df = pd.read_csv("backend/data/training_data.csv")
model = SentenceTransformer('backend/nlp-model/nlp-recipe-finder')

recipe_embeddings = torch.load("backend/model/recipe_embeddings.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_recipe_recommendations(input):

# Sample query
    query_embedding = model.encode(input, convert_to_tensor=True)

# Similarity ranking
    sims = cosine_similarity(query_embedding, recipe_embeddings)
    top_k = torch.topk(sims, k=5)
    top_indices = top_k.indices.tolist()

# Return top results sorted by rating
    results = df.iloc[top_indices]

    return results[['recipe_id','name', 'rating']]
