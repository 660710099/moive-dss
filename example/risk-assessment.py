import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_risk_assessment(
    predicted_revenue_mean, 
    revenue_std_dev, 
    planned_budget, 
    budget_std_dev, 
    planned_marketing, 
    marketing_std_dev, 
    iterations=100000
):
    print(f"--- Running Monte Carlo Simulation ({iterations:,} iterations) ---")
    
    # 1. SIMULATE COSTS (Normal Distribution)
    # Budgets and marketing can go over or under, but usually center around the plan.
    simulated_budgets = np.random.normal(planned_budget, budget_std_dev, iterations)
    simulated_marketing = np.random.normal(planned_marketing, marketing_std_dev, iterations)
    
    # Ensure costs don't drop to unrealistic negative numbers
    simulated_budgets = np.maximum(simulated_budgets, planned_budget * 0.5)
    simulated_marketing = np.maximum(simulated_marketing, planned_marketing * 0.5)
    
    total_simulated_costs = simulated_budgets + simulated_marketing

    # 2. SIMULATE REVENUE (Log-Normal Distribution)
    # Box office has a hard floor at $0 but a massive long-tail upside.
    # We must convert our normal mean/std into log-normal parameters (mu and sigma).
    variance = revenue_std_dev ** 2
    mu = np.log(predicted_revenue_mean ** 2 / np.sqrt(variance + predicted_revenue_mean ** 2))
    sigma = np.sqrt(np.log(variance / (predicted_revenue_mean ** 2) + 1))
    
    simulated_revenues = np.random.lognormal(mu, sigma, iterations)

    # 3. CALCULATE ROI FOR EVERY SCENARIO
    rois = (simulated_revenues - total_simulated_costs) / total_simulated_costs

    # 4. EXTRACT RISK METRICS
    probability_of_profit = np.mean(rois > 0) * 100
    expected_roi = np.mean(rois) * 100
    
    # Value at Risk (VaR) - What happens in the worst 5% of scenarios?
    var_95 = np.percentile(rois, 5) * 100
    
    # Optimistic Case - What happens in the best 5% of scenarios?
    bull_case_95 = np.percentile(rois, 95) * 100

    print(f"\n--- Investor Risk Report ---")
    print(f"Probability of Profit (ROI > 0%): {probability_of_profit:.2f}%")
    print(f"Expected Mean ROI:                {expected_roi:.2f}%")
    print(f"Worst Case (Bottom 5% VaR):       {var_95:.2f}% ROI")
    print(f"Best Case (Top 5% Upside):        {bull_case_95:.2f}% ROI")

    return rois

# ==========================================
# EXECUTE THE SIMULATION
# ==========================================
# Let's assume these inputs come from our XGBoost prediction model and budget sheets
predicted_box_office = 120.0  # $120M predicted revenue
box_office_error = 35.0       # Model's historical margin of error (Std Dev)

prod_budget = 50.0            # $50M planned budget
prod_budget_risk = 5.0        # $5M standard deviation (overages)

marketing_spend = 30.0        # $30M planned P&A (Prints & Advertising)
marketing_risk = 2.0          # $2M standard deviation

# Run the simulation
roi_distribution = run_monte_carlo_risk_assessment(
    predicted_box_office, box_office_error,
    prod_budget, prod_budget_risk,
    marketing_spend, marketing_risk
)

# ==========================================
# VISUALIZE THE RISK PROFILE
# ==========================================
plt.figure(figsize=(10, 6))
# Plot the histogram of all 100,000 simulated realities
plt.hist(roi_distribution * 100, bins=100, color='#2c3e50', alpha=0.8, edgecolor='black')

# Add a vertical line for the Break-Even point
plt.axvline(0, color='#e74c3c', linestyle='dashed', linewidth=2, label='Break Even (0% ROI)')
# Add a line for the Expected Mean
plt.axvline(np.mean(roi_distribution) * 100, color='#27ae60', linestyle='dashed', linewidth=2, label='Expected ROI')

plt.title('Movie Investment Risk Profile (100,000 Simulated Scenarios)', fontsize=14)
plt.xlabel('Return on Investment (%)', fontsize=12)
plt.ylabel('Frequency of Occurrence', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
