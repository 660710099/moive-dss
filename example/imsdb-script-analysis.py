import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline

print("--- 1. Loading the Script Dataset ---")
# Pull the clean corpus of 1,172 movie scripts directly from Hugging Face
print("Downloading/Loading dataset (this may take a moment on the first run)...")
dataset = load_dataset("IsmaelMousa/movies", split="train")

# We will grab the very first script in the dataset as an example
sample_movie = dataset[0]
movie_name = sample_movie['Name']
full_script = sample_movie['Script']

print(f"\nSuccessfully loaded script for: {movie_name}")
print(f"Script length: {len(full_script):,} characters")

print("\n--- 2. Preprocessing & Chunking ---")
# NLP Transformer models like DistilBERT have a strict 512-token limit. 
# We cannot feed a 100-page script in all at once. We must break it into smaller 
# chronological chunks (e.g., ~2,000 characters each, representing roughly a page or a scene).
chunk_size = 2000
script_chunks = [full_script[i:i + chunk_size] for i in range(0, len(full_script), chunk_size)]

# To save processing time for this demonstration, we will only analyze the first 25 chunks (roughly Act 1)
chunks_to_analyze = script_chunks[:25]
print(f"Divided script into {len(script_chunks)} total chunks.")
print(f"Analyzing the first {len(chunks_to_analyze)} chunks (Act 1)...\n")

print("--- 3. Running NLP Sentiment Analysis ---")
# Load a pre-trained DistilBERT sentiment analyzer
# We use truncation=True to ensure we don't crash if a chunk slightly exceeds the token limit
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

emotional_arc = []

# Process the script chronologically
for i, chunk in enumerate(chunks_to_analyze):
    # The model reads the scene and returns a label (POSITIVE/NEGATIVE) and a confidence score
    result = sentiment_analyzer(chunk)[0]
    
    # Convert this into a mathematical score: POSITIVE is +score, NEGATIVE is -score
    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    emotional_arc.append(score)
    
    # Print real-time progress to the terminal
    print(f"Scene/Chunk {i+1:02d} | Label: {result['label']:<8} | Narrative Score: {score:+.2f}")

print("\n--- 4. Visualizing the Narrative Arc ---")
# Plot the emotional journey of the script using Matplotlib
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(emotional_arc) + 1), emotional_arc, marker='o', linestyle='-', color='#3498db', linewidth=2)

# Add a baseline to easily see when the script dips into tragedy/conflict vs triumph
plt.axhline(0, color='#e74c3c', linestyle='dashed', linewidth=1.5, label='Neutral Baseline')

plt.title(f"Emotional Narrative Arc (Act 1): {movie_name}", fontsize=14, fontweight='bold')
plt.xlabel("Script Timeline (Chronological Chunks)", fontsize=12)
plt.ylabel("Sentiment Score (-1.0 to 1.0)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Display the graph
plt.show()
