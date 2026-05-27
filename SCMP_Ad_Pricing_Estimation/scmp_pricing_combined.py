import pandas as pd
import matplotlib.pyplot as plt

def analyze_scmp_combined():
    # Adjust path if running from subfolder
    INPUT_CSV = "SCMP_Ad_Pricing_Estimation/ad_reclassified_4class.csv"

    try:
        df = pd.read_csv(INPUT_CSV)
        scmp = df[df['Newspaper'] == 'SCMP'].copy()
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    if scmp.empty:
        return print("Error: No SCMP data found.")

    scmp['Date'] = pd.to_datetime(scmp['Date'])
    
    # Rule-Based Pricing (SCMP 2023 ROP Tiers)
    def get_rate(pct):
        try: pct = float(pct)
        except: pct = 10.0
        if pct < 5.0: return 7500
        elif pct < 20.0: return 16500
        elif pct < 50.0: return 55000
        else: return 150000

    def calc_rev(row):
        base = get_rate(row['Ad_Size_Percent'])
        mult = 1.25 if row['Date'].dayofweek >= 5 else 1.0
        if row['Date'].month in [12, 1]: mult *= 1.1
        return base * mult

    scmp['Est_Revenue'] = scmp.apply(calc_rev, axis=1)

    # Summary Stats (Requested Format)
    total_ads = len(scmp)
    unique_days = scmp['Date'].nunique()
    total_rev = scmp['Est_Revenue'].sum()
    
    print("SCMP Revenue Estimation (Combined Gov+SOE)")
    print(f"Total SCMP Ads Detected: {total_ads}")
    print(f"Active Days in Dataset: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue: HKD ${total_rev/unique_days:,.0f}" if unique_days else "N/A")
    print(f"Average Revenue Per Ad: HKD ${total_rev/total_ads:,.0f}\n")

    # Period Assignment
    def get_period(d):
        if pd.Timestamp('2020-01-01') <= d < pd.Timestamp('2020-06-28'): return 'P1'
        if pd.Timestamp('2020-06-28') <= d < pd.Timestamp('2021-06-13'): return 'P2'
        if pd.Timestamp('2021-06-17') <= d < pd.Timestamp('2022-06-17'): return 'P3'
        return None

    scmp['Period'] = scmp['Date'].apply(get_period)
    plot_data = scmp[scmp['Period'].notna()].copy()
    
    # Combine Types: Gov + SOE -> Public Sector
    def combine_type(val):
        if pd.isna(val): return None
        s = str(val).strip().lower()
        if s in ['government', 'soe']:
            return 'Public Sector'
        if s in ['commercial']:
            return 'Commercial'
        if s in ['other non-commercial']:
            return 'Other Non-Commercial'
        return 'Other Non-Commercial'

    plot_data['Type_Combined'] = plot_data['Type'].apply(combine_type)
    
    target_types = ['Commercial', 'Public Sector', 'Other Non-Commercial']

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}

    for i, ad_type in enumerate(target_types):
        ax = axes[i]
        data = plot_data[plot_data['Type_Combined'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type)
            continue

        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors[i])
        
        ax.set_title(f'{ad_type} Avg Revenue')
        ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('scmp_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved as 'scmp_combined_revenue.png'")

if __name__ == "__main__":
    analyze_scmp_combined()