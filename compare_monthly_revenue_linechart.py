import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ==================== CONFIGURATION ====================
INPUT_CSV = "Datasets/ad_reclassified_4class.csv"

TARGET_PAPERS = ['am730', 'scmp', 'hkcd', 'headlinedaily', 'the_standard']

# Historical FX Rates (RMB to HKD)
def get_fx_rate(year):
    if year < 2015: return 1.23
    elif year < 2019: return 1.20
    elif year < 2022: return 1.12
    elif year < 2025: return 1.08
    else: return 1.07

RATES = {
    'am730': {
        'full': {'color': 284000, 'red': 253000, 'bw': 234000},
        'double_full': {'color': 624800, 'red': 556600, 'bw': 514800},
        'junior': {'color': 184000, 'red': 165000, 'bw': 154000},
        'half': {'color': 156000, 'red': 142000, 'bw': 130000},
        'quarter': {'color': 86000, 'red': 78000, 'bw': 69000},
        'sixth': {'color': 58000, 'red': 53000, 'bw': 47000},
        'small': {'color': 20000, 'red': 15000, 'bw': 10000}
    },
    'scmp': {
        'full': {'color': 350000, 'red': 250000, 'bw': 200000},
        'junior': {'color': 200000, 'red': 150000, 'bw': 120000},
        'half': {'color': 175000, 'red': 125000, 'bw': 100000},
        'quarter': {'color': 90000, 'red': 65000, 'bw': 50000},
        'sixth': {'color': 50000, 'red': 35000, 'bw': 25000},
        'small': {'color': 15000, 'red': 10000, 'bw': 8000}
    },
    'hkcd': {
        'full': {'color': 160000, 'red': 110000, 'bw': 90000},
        'junior': {'color': 81800, 'red': 65800, 'bw': 50000},
        'half': {'color': 78000, 'red': 53800, 'bw': 40000},
        'quarter': {'color': 38000, 'red': 26000, 'bw': 20000},
        'sixth': {'color': 19000, 'red': 13000, 'bw': 10000},
        'small': {'color': 6400, 'red': 4500, 'bw': 3100}
    },
    'headlinedaily': {
        'full': {'color': 460000, 'red': 331000, 'bw': 285000},
        'junior': {'color': 294000, 'red': 209000, 'bw': 181000},
        'half': {'color': 241000, 'red': 175000, 'bw': 150000},
        'quarter': {'color': 120000, 'red': 88000, 'bw': 75000},
        'sixth': {'color': 85000, 'red': 59000, 'bw': 54000},
        'small': {'color': 30000, 'red': 20000, 'bw': 15000}
    },
    'the_standard': {
        'full': {'color': 156000, 'red': 109000, 'bw': 96000},
        'junior': {'color': 85000, 'red': 57000, 'bw': 53000},
        'half': {'color': 85000, 'red': 57000, 'bw': 53000},
        'quarter': {'color': 46000, 'red': 33000, 'bw': 27000},
        'sixth': {'color': 29000, 'red': 23000, 'bw': 17000},
        'small': {'color': 10000, 'red': 8000, 'bw': 6000}
    }
}

def calculate_revenue(row):
    newspaper = row['Newspaper'].lower() if pd.notna(row['Newspaper']) else ""
    size_pct = row.get('Ad_Size_Percent', 0)
    color = str(row.get('Color', 'full_color')).lower()
    ad_type = str(row.get('Type', '')).lower()
    
    # Smart Color Defaulting
    if 'bw' in color and 'red' not in color:
        color_type = 'bw'
    elif 'red' in color or 'spot' in color:
        color_type = 'red'
    elif pd.notna(color) and color != "":
        color_type = 'color'
    else:
        if 'government' in ad_type or 'other non-commercial' in ad_type:
            color_type = 'bw'
        else:
            color_type = 'color'

    # Size Mapping
    if newspaper == 'am730':
        if size_pct >= 150: spec = 'double_full'
        elif size_pct >= 95: spec = 'full'
        elif size_pct >= 70: spec = 'junior'
        elif size_pct >= 50: spec = 'half'
        elif size_pct >= 25: spec = 'quarter'
        elif size_pct >= 12: spec = 'sixth'
        else: spec = 'small'
    else:
        if size_pct >= 95: spec = 'full'
        elif size_pct >= 70: spec = 'junior'
        elif size_pct >= 50: spec = 'half'
        elif size_pct >= 25: spec = 'quarter'
        elif size_pct >= 12: spec = 'sixth'
        else: spec = 'small'

    try:
        price = RATES[newspaper][spec][color_type]
    except KeyError:
        try:
            price = RATES[newspaper]['small']['color']
        except:
            price = 0
    return price

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    df['Newspaper_Lower'] = df['Newspaper'].str.lower().str.strip()
    df_filtered = df[df['Newspaper_Lower'].isin(TARGET_PAPERS)].copy()
    
    if df_filtered.empty:
        return print("No data found for target newspapers.")

    # Calculate Base Revenue (Local Currency)
    df_filtered['Est_Revenue_Local'] = df_filtered.apply(calculate_revenue, axis=1)
    
    # Parse Date to get Year for FX Conversion
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Date'])
    df_filtered['Year'] = df_filtered['Date'].dt.year
    
    # Apply Dynamic FX Rate ONLY for HKCD
    def convert_to_hkd(row):
        if row['Newspaper_Lower'] == 'hkcd':
            fx = get_fx_rate(row['Year'])
            return row['Est_Revenue_Local'] * fx
        else:
            return row['Est_Revenue_Local'] # Already HKD

    df_filtered['Est_Revenue_HKD'] = df_filtered.apply(convert_to_hkd, axis=1)

    df_filtered['Month'] = df_filtered['Date'].dt.to_period('M')

    monthly_rev = df_filtered.groupby(['Newspaper_Lower', 'Month'])['Est_Revenue_HKD'].sum().reset_index()
    pivot_table = monthly_rev.pivot(index='Month', columns='Newspaper_Lower', values='Est_Revenue_HKD').fillna(0)
    pivot_table_smooth = pivot_table.rolling(window=3, min_periods=1).mean()
    pivot_table_smooth.index = pivot_table_smooth.index.astype(str)

    # ==================== PLOTTING ====================
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    colors = {
        'am730': '#1f77b4',
        'scmp': '#ff7f0e',
        'hkcd': '#2ca02c',
        'headlinedaily': '#d62728',
        'the_standard': '#9467bd'
    }

    for paper in TARGET_PAPERS:
        if paper in pivot_table_smooth.columns:
            ax.plot(pivot_table_smooth.index, pivot_table_smooth[paper], 
                    label=paper.replace('_', ' ').title(), 
                    color=colors.get(paper, 'gray'),
                    linewidth=2.5, marker='') 

    # Event Lines
    event_date_1_str = '2020-06'
    event_date_2_str = '2021-06'
    xticks = ax.get_xticks()
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]

    try:
        pos1 = xticklabels.index(event_date_1_str)
        pos2 = xticklabels.index(event_date_2_str)
    except ValueError:
        pos1, pos2 = None, None

    if pos1 is not None:
        ax.axvline(x=xticks[pos1], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot([], [], color='black', linestyle='--', linewidth=1.5, label='NSL Enacted (Jun 2020)')

    if pos2 is not None:
        ax.axvline(x=xticks[pos2], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot([], [], color='red', linestyle='--', linewidth=1.5, label='Apple Daily Closed (Jun 2021)')

    ax.set_title('Monthly Ad Revenue Comparison (5 Newspapers) - Dynamic FX (HKD)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Estimated Revenue (HKD)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    tick_positions = list(range(0, len(pivot_table_smooth.index), 6))
    tick_labels = [pivot_table_smooth.index[i] for i in tick_positions if i < len(pivot_table_smooth.index)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=1, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monthly_revenue_dynamic_fx.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: monthly_revenue_dynamic_fx.png")

if __name__ == "__main__":
    main()