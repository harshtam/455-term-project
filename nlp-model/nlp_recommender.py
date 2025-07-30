import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity

# Load model

def get_recipe_recommendations(input):

    df = pd.read_csv("training_data.csv")
    model = SentenceTransformer('nlp-recipe-finder')

    recipe_embeddings = torch.load("recipe_embeddings.pt", map_location=torch.device('cpu'))

# Sample query
    query_embedding = model.encode(input, convert_to_tensor=True)

# Similarity ranking
    sims = cosine_similarity(query_embedding, recipe_embeddings)
    top_k = torch.topk(sims, k=10)
    top_indices = top_k.indices.tolist()

# Return top results sorted by rating
    results = df.iloc[top_indices]
    results = results.sort_values(by='rating', ascending=False)

    return results[['recipe_id','name', 'rating', ]]