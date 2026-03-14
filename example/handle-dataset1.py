import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("--- 1. Loading and Cleaning the Data ---")
# Load your dataset (replace 'movies_dataset.csv' with your actual file name)
# df = pd.read_csv('movies_dataset.csv')

# For the sake of this script running out-of-the-box, here is a mock dataframe 
# structured exactly like your first dataset:
data = {
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'release_date': ['2010-05-12', '2012-07-20', '2015-06-15', '2018-11-20', '2022-05-05'],
    'budget': [50000000, 150000000, 20000000, 200000000, 40000000],
    'revenue': [120000000, 800000000, 45000000, 1000000000, 15000000],
    'runtime': [110, 145, 95, 130, 105],
    'popularity': [45.2, 112.5, 20.1, 150.8, 30.5],
    'vote_average': [6.5, 8.2, 7.0, 7.5, 5.1],
    'genres': ['Action, Sci-Fi', 'Action, Thriller', 'Horror', 'Action, Sci-Fi', 'Comedy'],
    'directors': ['Director X', 'Director Y', 'Director Z', 'Director Y', 'Director X'],
    'cast': ['Actor 1, Actor 2', 'Actor 3, Actor 4', 'Actor 5', 'Actor 3, Actor 1', 'Actor 6'],
    # Junk columns to drop
    'homepage': ['url1', 'url2', 'url3', 'url4', 'url5'],
    'poster_path': ['/img1', '/img2', '/img3', '/img4', '/img5']
}
df = pd.DataFrame(data)

# Drop useless columns immediately to save memory
columns_to_drop = ['title', 'homepage', 'poster_path'] # Add 'id', 'overview', etc. here
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Drop rows where we don't know the revenue or budget (cannot train on missing targets)
df = df.dropna(subset=['revenue', 'budget'])
df = df[(df['revenue'] > 1000) & (df['budget'] > 1000)] # Remove weird $1 placeholder budgets

print("\n--- 2. Time-Series Engineering ---")
# Convert release_date to datetime and strictly sort it. 
# THIS IS CRITICAL TO PREVENT DATA LEAKAGE!
df['release_date'] = pd.to_datetime(df['release_date'])
df = df.sort_values('release_date').reset_index(drop=True)

df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df = df.drop('release_date', axis=1)

print("\n--- 3. Target Encoding (Names to Numbers) ---")
# Extract the primary director and lead actor (first name in the list)
df['primary_director'] = df['directors'].apply(lambda x: str(x).split(',')[0].strip())
df['lead_actor'] = df['cast'].apply(lambda x: str(x).split(',')[0].strip())

# Calculate Historical Expanding Mean for Directors
df['director_hist_rev'] = df.groupby('primary_director')['revenue'].transform(
    lambda x: x.expanding().mean().shift()
)

# Calculate Historical Expanding Mean for Lead Actors
df['actor_hist_rev'] = df.groupby('lead_actor')['revenue'].transform(
    lambda x: x.expanding().mean().shift()
)

# Solve the "Cold Start Problem" for first-timers
global_median_rev = df['revenue'].median()

# Add our Boolean flags so XGBoost knows they are newcomers
df['is_debut_director'] = df['director_hist_rev'].isna().astype(int)
df['is_debut_actor'] = df['actor_hist_rev'].isna().astype(int)

# Fill the missing historical values with the global median
df['director_hist_rev'] = df['director_hist_rev'].fillna(global_median_rev)
df['actor_hist_rev'] = df['actor_hist_rev'].fillna(global_median_rev)

# Drop the text columns now that we have the math
df = df.drop(columns=['directors', 'cast', 'primary_director', 'lead_actor'])

print("\n--- 4. Categorical Engineering (One-Hot Encoding) ---")
# A simple way to one-hot encode comma-separated genres
genres_dummies = df['genres'].str.get_dummies(sep=', ')
# Let's only keep the most popular genres to avoid adding 100 tiny columns
top_genres = genres_dummies.sum().sort_values(ascending=False).head(10).index
genres_dummies = genres_dummies[top_genres]
genres_dummies = genres_dummies.add_prefix('genre_')

df = pd.concat([df, genres_dummies], axis=1)
df = df.drop('genres', axis=1)

print("\n--- 5. Training the XGBoost Model ---")
# Separate Features (X) and Target (y)
X = df.drop('revenue', axis=1)
y = df['revenue']

# Time-based split: Train on older movies, test on the newest 20%
# This simulates how the model will be used in reality
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Initialize and train the model
# (Parameters tuned for a mix of budget and historical features)
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

print("\n--- 6. Evaluating Performance ---")
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Test Set R-Squared: {r2:.2f} (1.0 is perfect)")
print(f"Mean Absolute Error: ${mae:,.2f} off per prediction")

print("\nTop 5 Most Important Features:")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance.head(5).to_string(index=False))

# NOTE: In a real project, you would save this trained model to use in your backend:
# model.save_model("xgboost_box_office_model.json")
