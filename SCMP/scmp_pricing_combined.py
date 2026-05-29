import pandas as pd
import matplotlib.pyplot as plt

def analyze_scmp_final():
    # ==================== CONFIGURATION ====================
    INPUT_CSV = "Datasets/ad_reclassified_4class.csv"

    # Milestones
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")

    try:
        df = pd.read_csv(INPUT_CSV)
        scmp = df[df['Newspaper'] == 'SCMP'].copy()
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    if scmp.empty or 'Type' not in scmp.columns:
        return print("Error: Invalid data or missing 'Type' column.")

    # Preprocess
    scmp['Date'] = pd.to_datetime(scmp['Date'])
    
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
        mult = 1.25 if row['Date'].dayofweek >= 5 else 1.0  # Weekend +25%
        if row['Date'].month in [12, 1]: mult *= 1.1        # Holiday +10%
        return base * mult

    scmp['Est_Revenue'] = scmp.apply(calc_rev, axis=1)

    # Summary Stats (Exact Format Requested)
    total_ads = len(scmp)
    unique_days = scmp['Date'].nunique()
    total_rev = scmp['Est_Revenue'].sum()
    avg_daily = total_rev / unique_days if unique_days else 0
    avg_per_ad = total_rev / total_ads if total_ads > 0 else 0
    
    print("SCMP Revenue Estimation")
    print(f"Total SCMP Ads Detected: {total_ads}")
    print(f"Active Days in Dataset: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue:   HKD ${avg_daily:,.0f}")
    print(f"Average Revenue Per Ad:  HKD ${avg_per_ad:,.0f}\n")

    # Period Assignment
    def assign_period(d):
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    scmp['Period'] = scmp['Date'].apply(assign_period)
    plot_data = scmp[scmp['Period'].notna()].copy()
    
    # Normalize Types for 4-Class
    plot_data['Type_Clean'] = plot_data['Type'].str.lower().str.strip()
    target_types_4 = ['commercial', 'government', 'soe', 'other non-commercial']
    plot_data_4 = plot_data[plot_data['Type_Clean'].isin(target_types_4)].copy()

    # Combine Types for 3-Class (Gov + SOE -> Public Sector)
    def combine_type(val):
        if pd.isna(val): return None
        s = str(val).strip().lower()
        if s in ['government', 'soe']:
            return 'Public Sector'
        if s in ['commercial']:
            return 'Commercial'
        return 'Other Non-Commercial'

    plot_data['Type_Combined'] = plot_data['Type'].apply(combine_type)
    target_types_3 = ['Commercial', 'Public Sector', 'Other Non-Commercial']

    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}
    colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_3 = ['#1f77b4', '#ff7f0e', '#2ca02c']
    display_map_4 = {t: t.replace('_', ' ').title() for t in target_types_4}

    if plot_data_4.empty:
        return print("No data for specified types/periods.")

    # --- PLOT 1: 4-CLASS (1 ROW) ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 6))
    
    for i, ad_type in enumerate(target_types_4):
        ax = axes1[i]
        data = plot_data_4[plot_data_4['Type_Clean'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_map_4[ad_type])
            ax.set_xticks([]); ax.set_yticks([])
            continue

        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_4[i])
        
        ax.set_title(f'{display_map_4[ad_type]} Avg Revenue (4-Class)')
        ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('SCMP/scmp_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: scmp_4class_revenue.png")

    # --- PLOT 2: 3-CLASS (COMBINED) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, ad_type in enumerate(target_types_3):
        ax = axes2[i]
        data = plot_data[plot_data['Type_Combined'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_3[i])
        
        ax.set_title(f'{ad_type} Avg Revenue (Combined)')
        ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('SCMP/scmp_3class_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: scmp_3class_combined_revenue.png")

if __name__ == "__main__":
    analyze_scmp_final()