import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# ==========================================
# 1. LOAD THE TRAINED AI MODEL
# ==========================================
# We use @st.cache_resource so it only loads the model once when the app starts
@st.cache_resource
def load_ai_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_box_office_model.json")
    return model

model = load_ai_model()

# ==========================================
# 2. BUILD THE USER INTERFACE
# ==========================================
st.set_page_config(page_title="Movie Investment DSS", layout="wide")
st.title("🎬 Greenlight: AI Box Office Predictor")
st.markdown("Enter the proposed movie details below to forecast its global box office revenue.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial & Metadata")
    budget = st.number_input("Production Budget ($)", min_value=1000000, max_value=500000000, value=50000000, step=1000000)
    runtime = st.slider("Target Runtime (Minutes)", min_value=80, max_value=200, value=110)
    popularity = st.slider("Expected Pre-Release Hype (TMDB Popularity Metric)", 1.0, 200.0, 45.0)
    
    # NEW: Added Release Date inputs
    col1_a, col1_b = st.columns(2)
    with col1_a:
        release_year = st.number_input("Release Year", min_value=2024, max_value=2035, value=2025)
    with col1_b:
        release_month = st.selectbox("Release Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=6) # Default to July
    
    # UPDATED: Full list of genres matching the training data perfectly
    all_genres = [
        "Action", "Adventure", "Comedy", "Crime", "Drama", "Family", 
        "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction", "Thriller"
    ]
    genre_selected = st.selectbox("Primary Genre", all_genres)

with col2:
    st.subheader("Talent Track Record")
    st.markdown("*Note: In a full app, these would auto-fill from a database.*")
    
    is_debut_director = st.checkbox("Is this the Director's first movie?")
    director_hist_rev = 20000000 if is_debut_director else st.number_input("Director's Historical Avg Box Office ($)", value=100000000)
        
    is_debut_actor = st.checkbox("Is this the Lead Actor's first major movie?")
    actor_hist_rev = 20000000 if is_debut_actor else st.number_input("Lead Actor's Historical Avg Box Office ($)", value=80000000)

# ==========================================
# 3. TRANSLATE UI TO MATH (STRICT FEATURE MATCHING)
# ==========================================
# This dictionary perfectly matches the order and names from your error message!
input_data = {
    'runtime': [runtime],
    'budget': [budget],
    'popularity': [popularity],
    'release_year': [release_year],
    'release_month': [release_month],
    'director_hist_rev': [director_hist_rev],
    'actor_hist_rev': [actor_hist_rev],
    'is_debut_director': [1 if is_debut_director else 0],
    'is_debut_actor': [1 if is_debut_actor else 0],
    
    # Initialize all possible genres to 0
    'genre_Drama': [0],
    'genre_Comedy': [0],
    'genre_Action': [0],
    'genre_Thriller': [0],
    'genre_Romance': [0],
    'genre_Adventure': [0],
    'genre_Crime': [0],
    'genre_Horror': [0],
    'genre_Family': [0],
    'genre_Science Fiction': [0],
    'genre_Fantasy': [0],
    'genre_Mystery': [0]
}

# Turn the selected genre into a 1 (One-Hot Encoding)
genre_key = f"genre_{genre_selected}"
if genre_key in input_data:
    input_data[genre_key] = [1]

# Convert to a Pandas DataFrame
# We make sure the columns are in the EXACT order the model expects
expected_columns = list(input_data.keys())
X_predict = pd.DataFrame(input_data)[expected_columns]

# ==========================================
# 4. RUN PREDICTION & DISPLAY RESULTS
# ==========================================
st.markdown("---")
if st.button("RUN PREDICTIVE ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("AI is analyzing historical data..."):
        try:
            # Predict
            prediction = model.predict(X_predict)[0]
            
            # Prevent negative numbers (XGBoost can sometimes guess below 0 for terrible movies)
            final_revenue = max(prediction, 0)
            roi = ((final_revenue - budget) / budget) * 100
            
            # Display Results
            st.success("Analysis Complete!")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted Global Box Office", f"${final_revenue:,.0f}")
            res_col2.metric("Projected ROI", f"{roi:.1f}%")
            
            if roi > 20:
                res_col3.success("🟢 RECOMMENDATION: GREENLIGHT")
            elif -10 <= roi <= 20:
                res_col3.warning("🟡 RECOMMENDATION: REWRITE / CUT BUDGET")
            else:
                res_col3.error("🔴 RECOMMENDATION: PASS")
                
        except ValueError as e:
            st.error(f"Feature Mismatch Error: Your model was trained on different columns than this dashboard provides. Make sure your genre_ columns match exactly. Error details: {e}")
