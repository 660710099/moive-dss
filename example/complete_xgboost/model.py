import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

print("--- 1. Loading and Cleaning the Data ---")
# Define the exact columns we need so we don't waste memory on text overviews or image URLs
columns_we_need = [
    'budget', 'genres', 'release_date', 'revenue', 
    'runtime', 'directors', 'cast', 'popularity'
]

# Load the CSV
try:
    df = pd.read_csv('dataset/TMDB_IMDB_Movies_Dataset.csv', usecols=columns_we_need, low_memory=False)
    print(f"Successfully loaded {len(df)} raw movies.")
except FileNotFoundError:
    print("ERROR: Could not find 'movies_dataset.csv'. Please check the file name.")
    exit()

# Drop rows where we are missing the crucial target variable or budget
df = df.dropna(subset=['revenue', 'budget', 'release_date'])
print(f"Movies remaining after removing NaN rows: {len(df)}")

# Filter out placeholder data (e.g., movies listed with a $1 budget or $0 revenue)
df = df[(df['revenue'] > 10000) & (df['budget'] > 10000)]
print(f"Movies remaining after cleaning missing/placeholder financials: {len(df)}")

print("\n--- 2. Time-Series Engineering ---")
# Convert release_date to datetime and STRICTLY SORT IT. 
# This prevents time-travel data leakage during target encoding.
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date']) # Drop any rows where the date was completely invalid
df = df.sort_values('release_date').reset_index(drop=True)

df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df = df.drop('release_date', axis=1)

print("\n--- 3. Target Encoding (Names to Numbers) ---")
# 1. Extract the names
df['primary_director'] = df['directors'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
df['lead_actor'] = df['cast'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()

# 2. Calculate Historical Expanding Mean
df['director_hist_rev'] = df.groupby('primary_director')['revenue'].transform(
    lambda x: x.expanding().mean().shift()
)
df['actor_hist_rev'] = df.groupby('lead_actor')['revenue'].transform(
    lambda x: x.expanding().mean().shift()
)

# ==========================================
# 🚨 THE CRITICAL FIX YOU JUST IDENTIFIED 🚨
# ==========================================
# We must manually destroy the fake track record Pandas built for the "Unknown" mega-director.
# We overwrite their historical revenue with NaN (Not a Number) so they are treated as newcomers.
import numpy as np
df.loc[df['primary_director'] == 'Unknown', 'director_hist_rev'] = np.nan
df.loc[df['lead_actor'] == 'Unknown', 'actor_hist_rev'] = np.nan

# 3. Solve the "Cold Start Problem"
global_median_rev = df['revenue'].median()

# Because we forced 'Unknown' to NaN above, this flag will correctly mark them all as 1 (Debut)
df['is_debut_director'] = df['director_hist_rev'].isna().astype(int)
df['is_debut_actor'] = df['actor_hist_rev'].isna().astype(int)

# Fill the missing historical values with the global median
df['director_hist_rev'] = df['director_hist_rev'].fillna(global_median_rev)
df['actor_hist_rev'] = df['actor_hist_rev'].fillna(global_median_rev)

# Drop the text columns
df = df.drop(columns=['directors', 'cast', 'primary_director', 'lead_actor'])

# print("\n--- 3. Target Encoding (Names to Numbers) ---")
# # Extract the primary director and lead actor (assuming comma-separated lists)

# # 1. Fill any blank cells with 'Unknown' so the math doesn't crash on NaNs
# df['directors'] = df['directors'].fillna('Unknown')
# df['cast'] = df['cast'].fillna('Unknown')

# # 2. Use the Pandas native .str accessor (much safer and faster than apply/lambda)
# df['primary_director'] = df['directors'].str.split(',').str[0].str.strip()
# df['lead_actor'] = df['cast'].str.split(',').str[0].str.strip()

# # Calculate Historical Expanding Mean (shifted to prevent the movie predicting itself)
# df['director_hist_rev'] = df.groupby('primary_director')['revenue'].transform(
#     lambda x: x.expanding().mean().shift()
# )
# df['actor_hist_rev'] = df.groupby('lead_actor')['revenue'].transform(
#     lambda x: x.expanding().mean().shift()
# )

# # Solve the "Cold Start Problem" for first-timers
# global_median_rev = df['revenue'].median()

# # Add Boolean flags so XGBoost knows they are newcomers (1 = debut, 0 = veteran)
# df['is_debut_director'] = df['director_hist_rev'].isna().astype(int)
# df['is_debut_actor'] = df['actor_hist_rev'].isna().astype(int)

# # Fill the missing historical values with the global median
# df['director_hist_rev'] = df['director_hist_rev'].fillna(global_median_rev)
# df['actor_hist_rev'] = df['actor_hist_rev'].fillna(global_median_rev)

# # Drop the text columns now that we have the math
# df = df.drop(columns=['directors', 'cast', 'primary_director', 'lead_actor'])

print("\n--- 4. Categorical Engineering (One-Hot Encoding Genres) ---")
# Split the comma-separated genres and create binary columns
genres_dummies = df['genres'].astype(str).str.get_dummies(sep=', ')

# Keep only the top 12 most common genres to prevent creating 100 useless tiny columns
top_genres = genres_dummies.sum().sort_values(ascending=False).head(12).index
genres_dummies = genres_dummies[top_genres]
genres_dummies = genres_dummies.add_prefix('genre_')

df = pd.concat([df, genres_dummies], axis=1)
df = df.drop('genres', axis=1)

# Ensure everything is strictly numeric before feeding to XGBoost
df = df.apply(pd.to_numeric, errors='coerce').dropna()

print("\n--- 5. Training the XGBoost Model ---")
# Separate Features (X) and Target (y)
X = df.drop('revenue', axis=1)
y = df['revenue']

# Time-based split: Train on the oldest 80% of movies, test on the newest 20%
# This perfectly simulates how an investor uses the model in the real world
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training on {len(X_train)} historical movies...")
print(f"Testing on {len(X_test)} newer movies...")

# Initialize and train the model
model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

print("\n--- 6. Evaluating Performance ---")
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Test Set R-Squared: {r2:.2f} (1.0 is a perfect oracle)")
print(f"Mean Absolute Error: ${mae:,.2f} off per prediction")

print("\nTop 5 Most Important Features driving Box Office Revenue:")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance.head(5).to_string(index=False))

# SAVE THE MODEL FOR THE DASHBOARD
model.save_model("xgboost_box_office_model.json")
print("\nSUCCESS: Model trained and saved as 'xgboost_box_office_model.json'")
