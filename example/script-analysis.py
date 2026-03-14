import numpy as np
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. LOAD THE DATA (Simulated Script Chunks)
# ==========================================
# In reality, you would use a library like PyPDF2 to extract text from a script
# and split it into chunks of ~400 words (representing scenes or pages).

new_script_chunks = [
    "The hero wakes up in a dark, oppressive room. He has no memory of how he got there. Panic sets in.", # Act 1: The Hook
    "He discovers a hidden door and escapes into a vibrant, futuristic city. He meets a group of rebels who offer him hope.", # Act 2: Rising Action
    "A massive battle ensues. The hero sacrifices himself to save the city. The rebels mourn, but the city is free." # Act 3: Resolution
]

# A benchmark "Hit Movie" script we want to compare against (e.g., The Matrix)
hit_movie_chunks = [
    "A hacker realizes his world is an illusion. He is chased by terrifying agents.",
    "He is rescued, learns the truth about the simulation, and begins intense training.",
    "He returns to the simulation, fights the agents, and realizes his true power to save humanity."
]

# ==========================================
# 2. EMOTIONAL ARC ANALYSIS (Sentiment extraction)
# ==========================================
print("--- 1. Generating Emotional Arc ---")
# Load a pre-trained sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def analyze_arc(script_chunks):
    arc = []
    for i, chunk in enumerate(script_chunks):
        result = sentiment_analyzer(chunk)[0]
        # Convert sentiment to a numerical score: Positive = +Score, Negative = -Score
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        arc.append(score)
        print(f"Act {i+1} Sentiment: {score:.2f} ({result['label']})")
    return arc

print("New Script Arc:")
new_script_arc = analyze_arc(new_script_chunks)

# ==========================================
# 3. VECTOR EMBEDDINGS & SIMILARITY SCORING
# ==========================================
print("\n--- 2. Calculating Narrative Similarity ---")
# Load a Sentence Transformer model to convert text into mathematical vectors
# 'all-MiniLM-L6-v2' is small, fast, and excellent for semantic similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Combine the chunks into a single narrative summary for embedding
new_script_text = " ".join(new_script_chunks)
hit_movie_text = " ".join(hit_movie_chunks)

# Generate high-dimensional vector embeddings
new_script_vector = embedder.encode([new_script_text])
hit_movie_vector = embedder.encode([hit_movie_text])

# Calculate Cosine Similarity
similarity_score = cosine_similarity(new_script_vector, hit_movie_vector)[0][0]

print(f"Narrative Similarity to Benchmark Hit: {similarity_score * 100:.2f}%")

if similarity_score > 0.75:
    print("System Assessment: High structural and thematic alignment with historically profitable IP.")
else:
    print("System Assessment: Script deviates significantly from established successful narrative structures.")
