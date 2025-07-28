# Imports
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader



# Load your dataset
df = pd.read_csv("training_data.csv")

# Create training pairs
def create_training_pairs(df):
    queries, positives, negatives = [], [], []
    for i in range(len(df)):
        query = df.iloc[i]['tags']
        pos = df.iloc[i]['tags']
        neg_idx = random.choice(df[df['rating'] < 3.0].index)
        negative = df.loc[neg_idx]['tags']
        queries.append(query)
        positives.append(pos)
        negatives.append(negative)
    return queries, positives, negatives

queries, positives, negatives = create_training_pairs(df)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
train_examples = [
    InputExample(texts=[q, p], label=1.0) for q, p in zip(queries, positives)
] + [
    InputExample(texts=[q, n], label=0.0) for q, n in zip(queries, negatives)
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True
)

# Save the model in a folder
model.save("nlp-recipe-finder")