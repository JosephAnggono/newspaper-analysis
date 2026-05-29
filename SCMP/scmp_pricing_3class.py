import pandas as pd
import matplotlib.pyplot as plt

def analyze_scmp_revenue():
    # Load Data
    try:
        df = pd.read_csv('Datasets/Ad_det_final.csv')
        scmp = df[df['Newspaper'] == 'SCMP'].copy()
    except FileNotFoundError:
        return print("Error: File not found.")

    if scmp.empty or 'Classification' not in scmp.columns:
        return print("Error: Invalid data or missing columns.")

    # Preprocess
    scmp['Date'] = pd.to_datetime(scmp['Date'])
    scmp['DayOfWeek'] = scmp['Date'].dt.dayofweek
    scmp['Month'] = scmp['Date'].dt.month

    # Rule-Based Pricing (SCMP 2023 ROP Tiers)
    def get_rate(pct):
        try: pct = float(pct)
        except: pct = 10.0
        
        if pct < 5.0: return 7500      # Small Box
        elif pct < 20.0: return 16500  # Medium Box
        elif pct < 50.0: return 55000  # Large/Standard
        else: return 150000            # Premium/Full Page

    def calc_rev(row):
        base = get_rate(row['Ad_Size_Percent'])
        mult = 1.25 if row['DayOfWeek'] >= 5 else 1.0  # Weekend +25%
        if row['Month'] in [12, 1]: mult *= 1.1        # Holiday +10%
        return base * mult

    scmp['Est_Revenue'] = scmp.apply(calc_rev, axis=1)

    # Key Metrics
    total_ads = len(scmp)
    unique_days = scmp['Date'].nunique()
    total_rev = scmp['Est_Revenue'].sum()
    
    print("SCMP Revenue Estimation")
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
    
    # Normalize Classifications
    target_types = ['commercial', 'government', 'other_non_commercial']
    plot_data['Class_Low'] = plot_data['Classification'].str.lower().str.strip()
    plot_data = plot_data[plot_data['Class_Low'].isin(target_types)]

    if plot_data.empty:
        return print("No data for specified types/periods.")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}

    for i, ad_type in enumerate(target_types):
        ax = axes[i]
        data = plot_data[plot_data['Class_Low'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type.title())
            continue

        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors)
        
        ax.set_title(f'{ad_type.replace("_", " ").title()} Avg Revenue')
        ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('scmp_3class_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_scmp_revenue()