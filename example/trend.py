import pandas as pd
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import time

print("--- 1. Initializing Google Trends API ---")
# hl = host language, tz = timezone offset
pytrends = TrendReq(hl='en-US', tz=360)

# Let's say we are considering casting one of these three actors for our lead role
cast_options = ["Timothée Chalamet", "Tom Holland", "Chris Pratt"]

print(f"Fetching 5-year trend data for: {', '.join(cast_options)}...")

# Build the payload (look at the last 5 years of global search data)
# Note: Google Trends allows a maximum of 5 keywords per request.
pytrends.build_payload(kw_list=cast_options, timeframe='today 5-y', geo='')

# Extract the interest over time as a Pandas DataFrame
trend_data = pytrends.interest_over_time()

# Drop the 'isPartial' column which just indicates if the current week's data is incomplete
if 'isPartial' in trend_data.columns:
    trend_data = trend_data.drop('isPartial', axis=1)

print("\n--- 2. Calculating Star Power Momentum Score ---")
# To make this useful for our Machine Learning model, we need a single number.
# We will calculate "Momentum": The average search volume of the last 6 months 
# divided by the average search volume of the previous 4.5 years.

# Split the data into 'recent' (last 26 weeks) and 'historical'
recent_data = trend_data.tail(26)
historical_data = trend_data.iloc[:-26]

momentum_scores = {}

for actor in cast_options:
    recent_avg = recent_data[actor].mean()
    historical_avg = historical_data[actor].mean()
    
    # Avoid division by zero
    if historical_avg == 0:
        momentum = 1.0 
    else:
        momentum = recent_avg / historical_avg
        
    momentum_scores[actor] = momentum
    print(f"{actor}:")
    print(f"  - Historical Baseline: {historical_avg:.2f}")
    print(f"  - Recent 6-Month Avg:  {recent_avg:.2f}")
    print(f"  => Momentum Score:     {momentum:.2f}x")

print("\nSystem Recommendation:")
top_actor = max(momentum_scores, key=momentum_scores.get)
print(f"Casting {top_actor} provides the highest current trend momentum for the project.")

print("\n--- 3. Visualizing Star Power Over Time ---")
# Plotting the 5-year trend lines
plt.figure(figsize=(12, 6))

for actor in cast_options:
    # We apply a 4-week rolling average to smooth out the spiky weekly data
    plt.plot(trend_data.index, trend_data[actor].rolling(window=4).mean(), label=actor, linewidth=2)

plt.title("Star Power: 5-Year Search Interest (Rolling 4-Week Average)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Relative Search Volume (0-100)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
