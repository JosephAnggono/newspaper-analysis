import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def estimate_scmp_revenue():
    # 1. Load Data
    df = pd.read_csv('Ad_det_final.csv')
    scmp_df = df[df['Newspaper'] == 'SCMP'].copy()
    
    if scmp_df.empty:
        print("No SCMP data found.")
        return

    # 2. Prepare Dates
    scmp_df['Date'] = pd.to_datetime(scmp_df['Date'])
    scmp_df['DayOfWeek'] = scmp_df['Date'].dt.dayofweek # 0=Mon, ..., 6=Sun
    scmp_df['Month'] = scmp_df['Date'].dt.month

    # 3. Define Pricing Rules (Based on SCMP 2023 Rate Card - Main Position ROP)
    def get_base_rate(size_pct):
        """
        Maps Ad_Size_Percent to SCMP ROP Tiers.
        Rates are Full Colour HKD.
        """
        try:
            pct = float(size_pct)
        except (ValueError, TypeError):
            pct = 10.0  # We set default to Medium Box Ads if data is missing

        # Size 1: Small Box Ads (<5% of page) -> Avg ~$7,500
        if pct < 5.0:
            return 7500
        
        # Tier 2: Medium/Large Box Ads (5-20%) -> Avg ~$16,500
        elif pct < 20.0:
            return 16500
        
        # Tier 3: Standard Large Ads (20-50%) -> Avg ~$55,000
        elif pct < 50.0:
            return 55000
        
        # Tier 4: Premium/Full Page (>50%) -> Avg ~$150,000
        else:
            return 150000

    def calculate_revenue(row):
        base_rate = get_base_rate(row['Ad_Size_Percent'])
        
        # Weekend Multiplier (Sat/Sun +25%)
        multiplier = 1.25 if row['DayOfWeek'] >= 5 else 1.0
        
        # Holiday Season Premium (Dec/Jan +10%)
        if row['Month'] in [12, 1]:
            multiplier *= 1.1
            
        return base_rate * multiplier

    # Apply Calculation
    scmp_df['Est_Revenue_HKD'] = scmp_df.apply(calculate_revenue, axis=1)

    # 4. Calculate Totals
    total_ads = len(scmp_df)
    unique_days = scmp_df['Date'].nunique()
    total_revenue = scmp_df['Est_Revenue_HKD'].sum()
    
    avg_per_ad = total_revenue / total_ads if total_ads > 0 else 0
    avg_daily = total_revenue / unique_days if unique_days > 0 else 0

    # Print Results
    print(f"--- SCMP Ad Revenue Estimation (2023 Rates) ---")
    print(f"Total SCMP Ads Detected: {total_ads}")
    print(f"Active Days in Dataset:  {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_revenue:,.2f}")
    print(f"Average Daily Revenue:   HKD ${avg_daily:,.2f}")
    print(f"Average Revenue Per Ad:  HKD ${avg_per_ad:,.2f}\n")

    # 5. Plotting Graphs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Revenue by Size Tier
    # Create a temporary column for tier labels for plotting
    scmp_df['Size_Tier'] = pd.cut(scmp_df['Ad_Size_Percent'], 
                                  bins=[-1, 5, 20, 50, 100], 
                                  labels=['Small', 'Medium', 'Large', 'Premium'])
    
    tier_totals = scmp_df.groupby('Size_Tier')['Est_Revenue_HKD'].sum()
    axes[0].bar(tier_totals.index, tier_totals.values, color='steelblue')
    axes[0].set_title('Total Revenue by Ad Size Tier')
    axes[0].set_ylabel('Revenue (HKD)')

    # Plot 2: Avg Revenue by Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = scmp_df.groupby(scmp_df['Date'].dt.day_name())['Est_Revenue_HKD'].mean().reindex(day_order)
    axes[1].bar(daily_avg.index, daily_avg.values, color='coral')
    axes[1].set_title('Avg Revenue Per Ad by Day')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Monthly Trend
    monthly_rev = scmp_df.groupby('Month')['Est_Revenue_HKD'].sum()
    axes[2].plot(monthly_rev.index, monthly_rev.values, marker='o', color='green', linewidth=2)
    axes[2].set_title('Total Monthly Revenue Trend')
    axes[2].set_xlabel('Month')
    axes[2].set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig('scmp_revenue_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved as 'scmp_revenue_plots.png'")

if __name__ == "__main__":
    estimate_scmp_revenue()