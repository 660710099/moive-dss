import pandas as pd
import numpy as np

print("--- 1. Setting Up the Data ---")
# Creating a dummy dataset to demonstrate the math
data = {
    'title': ['Following', 'Memento', 'Insomnia', 'Batman Begins', 'The Prestige', 'The Dark Knight'],
    'release_date': ['1998-09-12', '2000-09-05', '2002-05-24', '2005-06-15', '2006-10-20', '2008-07-18'],
    'directors': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan'],
    'revenue': [240000, 39700000, 113700000, 373600000, 109700000, 1005000000]
}

df = pd.DataFrame(data)

# Ensure the release date is actually a datetime object, then sort chronologically
df['release_date'] = pd.to_datetime(df['release_date'])
df = df.sort_values('release_date').reset_index(drop=True)

print("Original Data (Notice the revenues):")
print(df[['title', 'release_date', 'revenue']])

print("\n--- 2. Calculating Historical Average ---")
# Step A: Group by the director
# Step B: Use expanding().mean() to calculate a running average over time
# Step C: Use shift() to push the average DOWN one row. 
# (This ensures the movie's own revenue isn't used to predict itself!)

df['director_historical_avg'] = df.groupby('directors')['revenue'].transform(
    lambda x: x.expanding().mean().shift()
)

# Fill the very first movie with the dataset's overall median revenue (or 0)
# because they have no track record yet.
global_median_revenue = df['revenue'].median()
df['director_historical_avg'] = df.groupby('directors')['director_historical_avg'].fillna(0)

print("\nEngineered Feature for XGBoost:")
print(df[['title', 'directors', 'revenue', 'director_historical_avg']])

print("\n--- 3. Let's look at the math for 'The Prestige' ---")
# The Prestige is index 4. The model should only know about the first 4 movies.
movies_before_prestige = df.iloc[0:4]['revenue'].tolist()
math_check = sum(movies_before_prestige) / len(movies_before_prestige)
print(f"Revenues before The Prestige: {movies_before_prestige}")
print(f"Calculated Average: ${math_check:,.2f}")
print(f"DataFrame Value:    ${df.loc[4, 'director_historical_avg']:,.2f}")
