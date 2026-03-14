import pandas as pd
import numpy as np
import xgboost as xgb
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("--- 1. Loading and Cleaning TMDB Data ---")
# Load the dataset
# Make sure 'tmdb_5000_movies.csv' is in your working directory
try:
    df = pd.read_csv('tmdb_5000_movies.csv')
except FileNotFoundError:
    print("Error: Please download 'tmdb_5000_movies.csv' from Kaggle and place it in this directory.")
    exit()

# Filter out movies with $0 budget or $0 revenue (these are usually straight-to-DVD or missing data)
df = df[(df['budget'] > 100000) & (df['revenue'] > 100000)].copy()

print(f"Usable movies after cleaning $0 records: {len(df)}")

print("\n--- 2. Feature Engineering ---")
# 2a. Parse the 'genres' column (it's stored as a stringified list of dictionaries)
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

df['genre_list'] = df['genres'].apply(extract_genres)

# Create a simple binary feature for the most popular commercial genres
top_genres = ['Action', 'Comedy', 'Drama', 'Science Fiction', 'Thriller']
for genre in top_genres:
    df[f'is_{genre}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)

# 2b. Extract seasonality from release_date
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_month'] = df['release_date'].dt.month
# Fill any missing months with the median
df['release_month'] = df['release_month'].fillna(df['release_month'].median())

# 2c. Select the final features for the model
# TMDB's 'popularity', 'vote_count', and 'vote_average' act as our proxy for "Star Power" and "Hype"
features = [
    'budget', 
    'popularity', 
    'runtime', 
    'vote_average', 
    'vote_count',
    'release_month',
    'is_Action', 
    'is_Comedy', 
    'is_Drama', 
    'is_Science Fiction', 
    'is_Thriller'
]

X = df[features]
y = df['revenue']

# Handle any remaining missing values (like missing runtimes)
X = X.fillna(X.median())

print("\n--- 3. Training the XGBoost Model ---")
# Split the historical data: 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
# We use log-transformed objective for revenue because it spans from thousands to billions
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=200, 
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8
)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate error margins
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model R-Squared (Accuracy Score): {r2:.2f}")
print(f"Mean Absolute Error: ${mae:,.2f}")

print("\n--- 4. Predicting a New Movie Pitch ---")
# Let's say we are pitching a highly anticipated $100M Summer Action Sci-Fi movie
new_movie_pitch = pd.DataFrame({
    'budget': [100000000],          # $100 Million
    'popularity': [150.0],          # Very high anticipated hype
    'runtime': [120.0],             # 2 hours long
    'vote_average': [7.5],          # Projected solid reviews
    'vote_count': [5000],           # Projected high engagement
    'release_month': [7],           # July (Summer Blockbuster Season)
    'is_Action': [1],               # Yes
    'is_Comedy': [0],               # No
    'is_Drama': [0],                # No
    'is_Science Fiction': [1],      # Yes
    'is_Thriller': [0]              # No
})

predicted_revenue = model.predict(new_movie_pitch)[0]
print(f"Predicted Global Box Office Revenue: ${predicted_revenue:,.2f}")
