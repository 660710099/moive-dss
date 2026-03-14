import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. CREATE SYNTHETIC HISTORICAL DATA
# ==========================================
# In a real DSS, you load this from your PostgreSQL database
np.random.seed(42)
num_records = 1000

data = {
    'budget_millions': np.random.uniform(5, 200, num_records),
    'star_power_index': np.random.uniform(1, 10, num_records), # 10 is an A-list star
    'director_score': np.random.uniform(1, 10, num_records),
    'marketing_spend_millions': np.random.uniform(1, 50, num_records),
    'is_sequel': np.random.choice([0, 1], num_records, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Simulate Revenue based on inputs + random noise
df['box_office_revenue'] = (
    df['budget_millions'] * 1.5 +
    df['star_power_index'] * 5 +
    df['director_score'] * 3 +
    df['marketing_spend_millions'] * 2 +
    df['is_sequel'] * 30 +
    np.random.normal(0, 20, num_records) # Adding market unpredictability
)

# ==========================================
# 2. TRAIN THE XGBOOST PREDICTION MODEL
# ==========================================
X = df.drop('box_office_revenue', axis=1)
y = df['box_office_revenue']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Mean Absolute Error: ${mae:.2f} Million")

# ==========================================
# 3. PREDICT A NEW MOVIE & RUN MONTE CARLO
# ==========================================
# Let's say an investor pitches a new Sci-Fi movie
new_movie = pd.DataFrame({
    'budget_millions': [50],
    'star_power_index': [8],  # Strong lead actor
    'director_score': [7],
    'marketing_spend_millions': [20],
    'is_sequel': [0]          # Original IP
})

# Get the baseline point prediction from our ML model
predicted_revenue = model.predict(new_movie)[0]
print(f"\nBaseline Revenue Prediction: ${predicted_revenue:.2f} Million")

# Monte Carlo Simulation (10,000 iterations to assess risk)
iterations = 10000
simulated_revenues = np.random.normal(predicted_revenue, mae, iterations)
simulated_costs = np.random.normal(new_movie['budget_millions'][0] + new_movie['marketing_spend_millions'][0], 5, iterations) # Assuming +/- 5M cost variance

# ROI Formula
# ROI = (Revenue - Cost) / Cost
rois = (simulated_revenues - simulated_costs) / simulated_costs

# Calculate Risk Metrics
probability_of_profit = np.mean(rois > 0) * 100
average_roi = np.mean(rois) * 100

print(f"--- Investment Risk Report ---")
print(f"Probability of Breaking Even/Profit: {probability_of_profit:.1f}%")
print(f"Expected Average ROI: {average_roi:.1f}%")

# (Optional) Plot the risk curve
plt.hist(rois, bins=50, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Break Even (0% ROI)')
plt.title('Monte Carlo Simulation: ROI Probability Distribution')
plt.xlabel('Return on Investment (ROI)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
