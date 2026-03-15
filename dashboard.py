import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# ==========================================
# 1. LOAD MODEL & DATA (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_ai_model():
    model = xgb.XGBRegressor()
    # Ensure this matches the exact name of your saved model file
    model.load_model("xgboost_box_office_model.json")
    return model

@st.cache_data
def load_historical_data():
    """Loads the CSV into the background to act as our talent database."""
    columns_we_need = ['revenue', 'directors', 'cast', 'popularity']
    try:
        df = pd.read_csv('dataset/TMDB_IMDB_Movies_Dataset.csv', usecols=columns_we_need, low_memory=False)
        df = df.dropna(subset=['revenue'])
        # Clean the text names exactly like we did in training
        df['primary_director'] = df['directors'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
        df['lead_actor'] = df['cast'].fillna('Unknown').astype(str).str.split(',').str[0].str.strip()
        return df
    except FileNotFoundError:
        st.error("ERROR: Could not find 'movies_dataset.csv'. Please ensure it is in the same folder.")
        return pd.DataFrame()

model = load_ai_model()
db = load_historical_data()

# Extract unique, sorted names for the dropdowns (ignoring the 'Unknown' placeholders)
unique_directors = sorted([str(d) for d in db['primary_director'].unique() if str(d) != 'Unknown'])
unique_actors = sorted([str(a) for a in db['lead_actor'].unique() if str(a) != 'Unknown'])

# Add a "Debut" option at the very top of the list
director_options = ["(Debut / Unknown)"] + unique_directors
actor_options = ["(Debut / Unknown)"] + unique_actors

# Calculate global fallbacks for the "Cold Start Problem"
GLOBAL_MEDIAN_REV = db['revenue'].median() if not db.empty else 20000000
GLOBAL_MEDIAN_POP = db['popularity'].median() if not db.empty else 10.0

# ==========================================
# 2. BUILD THE USER INTERFACE
# ==========================================
st.set_page_config(page_title="Movie Investment DSS", layout="wide")
st.title("🎬 Greenlight: AI Box Office Predictor")
st.markdown("Enter the proposed movie details. The AI will query the database to evaluate the talent and automatically calculate expected pre-release hype.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial & Metadata")
    budget = st.number_input("Production Budget ($)", min_value=1000000, max_value=500000000, value=50000000, step=1000000)
    runtime = st.slider("Target Runtime (Minutes)", min_value=80, max_value=200, value=110)
    
    col1_a, col1_b = st.columns(2)
    with col1_a:
        release_year = st.number_input("Release Year", min_value=2024, max_value=2035, value=2025)
    with col1_b:
        release_month = st.selectbox("Release Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=6)
    
    all_genres = [
        "Action", "Adventure", "Comedy", "Crime", "Drama", "Family", 
        "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"
    ]
    genre_selected = st.selectbox("Primary Genre", all_genres)

with col2:
    st.subheader("Talent Attachments")
    st.markdown("*Type names exactly as they appear in the TMDB database (e.g., Christopher Nolan, Tom Holland).*")
    
    director_input = st.selectbox("Primary Director Name", options=director_options)
    actor_input = st.selectbox("Lead Actor Name", options=actor_options)
    
# ==========================================
# 3. DYNAMIC DATABASE QUERY & CALCULATIONS
# ==========================================
dir_hist_rev, dir_pop, is_debut_director = GLOBAL_MEDIAN_REV, GLOBAL_MEDIAN_POP, 1
actor_hist_rev, actor_pop, is_debut_actor = GLOBAL_MEDIAN_REV, GLOBAL_MEDIAN_POP, 1

# DIRECTOR LOGIC WITH ZERO-DOLLAR SAFETY NET
if director_input != "(Debut / Unknown)" and not db.empty:
    dir_movies = db[db['primary_director'].str.lower() == director_input.lower()]
    if len(dir_movies) > 0:
        raw_avg = dir_movies['revenue'].mean()
        
        # Intercept the zero! If it's missing or effectively zero, treat them as unknown.
        if pd.isna(raw_avg) or raw_avg <= 10000:
            dir_hist_rev = GLOBAL_MEDIAN_REV
            is_debut_director = 1  
        else:
            dir_hist_rev = raw_avg
            is_debut_director = 0
            
        # We can still safely grab their popularity hype even if financial data is missing
        dir_pop = dir_movies['popularity'].mean()

# ACTOR LOGIC WITH ZERO-DOLLAR SAFETY NET
if actor_input != "(Debut / Unknown)" and not db.empty:
    actor_movies = db[db['lead_actor'].str.lower() == actor_input.lower()]
    if len(actor_movies) > 0:
        raw_avg = actor_movies['revenue'].mean()
        
        # Intercept the zero!
        if pd.isna(raw_avg) or raw_avg <= 10000:
            actor_hist_rev = GLOBAL_MEDIAN_REV
            is_debut_actor = 1
        else:
            actor_hist_rev = raw_avg
            is_debut_actor = 0
            
        actor_pop = actor_movies['popularity'].mean()

# Auto-Calculate Expected TMDB Hype (Popularity)
if is_debut_director == 0 and is_debut_actor == 0:
    expected_popularity = (dir_pop + actor_pop) / 2
elif is_debut_director == 0:
    expected_popularity = dir_pop
elif is_debut_actor == 0:
    expected_popularity = actor_pop
else:
    expected_popularity = GLOBAL_MEDIAN_POP

# Display the AI's internal calculations to the user for transparency
with col2:
    st.info(f"**AI Internal Calculation:**\n"
            f"- Director Track Record: {'Debut/Unknown' if is_debut_director else f'${dir_hist_rev:,.0f} Avg'}\n"
            f"- Actor Track Record: {'Debut/Unknown' if is_debut_actor else f'${actor_hist_rev:,.0f} Avg'}\n"
            f"- Auto-Calculated Pre-Release Hype Score: {expected_popularity:.1f}")

# ==========================================
# 4. TRANSLATE TO XGBOOST MATRIX
# ==========================================
input_data = {
    'runtime': [runtime],
    'budget': [budget],
    'popularity': [expected_popularity],  # <--- Using the dynamic calculation!
    'release_year': [release_year],
    'release_month': [release_month],
    'director_hist_rev': [dir_hist_rev],
    'actor_hist_rev': [actor_hist_rev],
    'is_debut_director': [is_debut_director],
    'is_debut_actor': [is_debut_actor],
    
    # Genres initialized to 0
    'genre_Drama': [0], 'genre_Comedy': [0], 'genre_Action': [0], 'genre_Thriller': [0],
    'genre_Romance': [0], 'genre_Adventure': [0], 'genre_Crime': [0], 'genre_Horror': [0],
    'genre_Family': [0], 'genre_Science Fiction': [0], 'genre_Fantasy': [0], 'genre_Mystery': [0]
}

# Apply One-Hot Encoding
genre_key = f"genre_{genre_selected}"
if genre_key in input_data:
    input_data[genre_key] = [1]

expected_columns = list(input_data.keys())
X_predict = pd.DataFrame(input_data)[expected_columns]

# ==========================================
# 5. EXECUTE PREDICTION
# ==========================================
st.markdown("---")
if st.button("RUN PREDICTIVE ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Executing XGBoost Math & Monte Carlo Risk Simulation..."):
        try:
            # 1. Run the Math
            prediction = model.predict(X_predict)[0]
            final_revenue = max(prediction, 0) # Prevent negative box office
            roi = ((final_revenue - budget) / budget) * 100
            
            # 2. Display Top-Level Metrics
            st.success("Analysis Complete!")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            res_col1.metric("Predicted Global Box Office", f"${final_revenue:,.0f}")
            res_col2.metric("Projected ROI", f"{roi:.1f}%")
            
            if roi > 20:
                res_col3.success("🟢 RECOMMENDATION: GREENLIGHT")
            elif -10 <= roi <= 20:
                res_col3.warning("🟡 RECOMMENDATION: REWRITE / NEGOTIATE BUDGET")
            else:
                res_col3.error("🔴 RECOMMENDATION: PASS")
                
            st.markdown("---")
            
            # 3. Build the Visual Bar Chart
            st.subheader("Financial Breakdown")
            
            # Create a simple DataFrame specifically for the chart
            chart_data = pd.DataFrame({
                "Category": ["Production Budget", "Predicted Box Office"],
                "Amount ($)": [budget, final_revenue]
            })
            
            # Set the Category as the index so Streamlit knows what to put on the X-axis
            chart_data = chart_data.set_index("Category")
            
            # Draw the chart!
            st.bar_chart(chart_data)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
