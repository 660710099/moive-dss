import numpy as np
import pandas as pd
import xgboost as xgb
from transformers import pipeline

# ==========================================
# 1. INITIALIZE THE ENGINES (Load once in production)
# ==========================================
print("Booting up the DSS Engines...")

# A. NLP Engine (Using a lightweight sentiment model for speed)
nlp_engine = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)

# B. Box Office Predictor (Simulating a pre-trained XGBoost model)
# In reality, you would load your saved TMDB model here using xgb.Booster()
# For this demo, we will train a quick dummy model so the code actually runs
X_dummy = pd.DataFrame({'budget': [50, 10, 200, 100], 'star_power': [5, 2, 9, 7], 'is_action': [1, 0, 1, 1]})
y_dummy = pd.Series([150, 15, 600, 250])
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10).fit(X_dummy, y_dummy)

# ==========================================
# 2. THE THREE CORE PIPELINE FUNCTIONS
# ==========================================

def run_model_1_script_analysis(script_text):
    """Reads the script and outputs a viability multiplier (0.5 to 1.5)"""
    print("\n-> [Model 1] Analyzing Script Narrative...")
    
    # Chunk the script to bypass the 512 token limit
    chunks = [script_text[i:i+1000] for i in range(0, len(script_text), 1000)]
    scores = []
    
    for chunk in chunks[:5]: # Analyze first 5 chunks for speed
        result = nlp_engine(chunk)[0]
        # Positive scenes add to the score, Negative scenes detract slightly
        score = result['score'] if result['label'] == 'POSITIVE' else (1.0 - result['score'])
        scores.append(score)
    
    # Calculate average script sentiment and map it to a multiplier
    avg_score = np.mean(scores)
    
    # A perfectly average script gets a 1.0. A great script gets a boost up to 1.5x.
    script_multiplier = 0.5 + avg_score 
    print(f"   [Result] Script Viability Multiplier: {script_multiplier:.2f}x")
    return script_multiplier

def run_model_2_box_office_predictor(budget, star_power, is_action, script_multiplier):
    """Predicts baseline revenue and adjusts it using the script score."""
    print("-> [Model 2] Predicting Global Box Office...")
    
    # Structure the investor's inputs for the XGBoost model
    features = pd.DataFrame({
        'budget': [budget], 
        'star_power': [star_power], 
        'is_action': [is_action]
    })
    
    # Get the raw financial prediction
    baseline_prediction = xgb_model.predict(features)[0]
    
    # FUSION MATcH: Adjust the prediction based on the NLP script analysis
    adjusted_prediction = baseline_prediction * script_multiplier
    print(f"   [Result] Adjusted Revenue Prediction: ${adjusted_prediction:.2f} Million")
    return adjusted_prediction

def run_model_3_risk_simulation(predicted_revenue, budget, marketing_spend):
    """Runs 10,000 Monte Carlo simulations to find the probability of profit."""
    print("-> [Model 3] Running Monte Carlo Risk Simulation...")
    
    iterations = 10000
    total_planned_cost = budget + marketing_spend
    
    # Simulate slightly variable costs (Normal distribution)
    sim_costs = np.random.normal(total_planned_cost, total_planned_cost * 0.1, iterations)
    
    # Simulate highly variable revenues (Log-normal distribution to account for massive hits/flops)
    variance = (predicted_revenue * 0.35) ** 2
    mu = np.log(predicted_revenue ** 2 / np.sqrt(variance + predicted_revenue ** 2))
    sigma = np.sqrt(np.log(variance / (predicted_revenue ** 2) + 1))
    sim_revenues = np.random.lognormal(mu, sigma, iterations)
    
    # Calculate ROI for all 10,000 realities
    rois = (sim_revenues - sim_costs) / sim_costs
    
    probability_of_profit = np.mean(rois > 0) * 100
    expected_roi = np.mean(rois) * 100
    
    print(f"   [Result] Win Probability: {probability_of_profit:.1f}% | Expected ROI: {expected_roi:.1f}%")
    return probability_of_profit, expected_roi

# ==========================================
# 3. THE MASTER "GREENLIGHT" FUNCTION
# ==========================================

def evaluate_movie_pitch(pitch_name, script_text, budget, marketing, star_power, is_action):
    print(f"\n==============================================")
    print(f"EVALUATING PITCH: {pitch_name.upper()}")
    print(f"==============================================")
    
    # Pipeline Step 1: Text to Math
    script_multiplier = run_model_1_script_analysis(script_text)
    
    # Pipeline Step 2: Financials + Text Math to Predicted Dollars
    predicted_revenue = run_model_2_box_office_predictor(budget, star_power, is_action, script_multiplier)
    
    # Pipeline Step 3: Predicted Dollars + Costs to Risk Metrics
    win_prob, expected_roi = run_model_3_risk_simulation(predicted_revenue, budget, marketing)
    
    # Pipeline Step 4: The Final Composite Score (0-100)
    # 60% weight on Win Probability, 40% weight on Expected ROI (capped at 100%)
    capped_roi = min(max(expected_roi, 0), 100)
    greenlight_score = (win_prob * 0.60) + (capped_roi * 0.40)
    
    print("\n----------------------------------------------")
    print(f"FINAL SYSTEM GREENLIGHT SCORE: {greenlight_score:.1f} / 100")
    if greenlight_score >= 70:
        print("RECOMMENDATION: 🟢 APPROVED FOR INVESTMENT")
    elif 45 <= greenlight_score < 70:
        print("RECOMMENDATION: 🟡 HOLD (REDUCE BUDGET OR REWRITE SCRIPT)")
    else:
        print("RECOMMENDATION: 🔴 REJECT PITCH")
    print("----------------------------------------------\n")

# ==========================================
# 4. TEST THE SYSTEM
# ==========================================
# A mock script summary (in reality, this would be the 120-page text)
sample_script = "The hero is trapped in a burning skyscraper. He fights his way out against impossible odds. It is thrilling, explosive, and ends in a massive victory for humanity."

evaluate_movie_pitch(
    pitch_name="Skyscraper Assault",
    script_text=sample_script,
    budget=60.0,            # $60M Production Budget
    marketing=30.0,         # $30M P&A Budget
    star_power=8.0,         # High A-list star power
    is_action=1             # Yes, it's an action movie
)
