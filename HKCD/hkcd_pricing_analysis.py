import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_hkcd_final():
    # ==================== CONFIGURATION ====================
    INPUT_CSV = "Datasets/ad_reclassified_4class.csv" 
    
    # Milestones
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")
    
    COLOR_FULL = "full_color"
    COLOR_BW_RED = "bw_red" 
    COLOR_BW = "bw"

    RATE_CARD_HKCD = {
        "full_page":          {COLOR_FULL: 160000, COLOR_BW_RED: 110000, COLOR_BW: None},
        "small_full_page":    {COLOR_FULL: 81800,  COLOR_BW_RED: 65800,  COLOR_BW: None},
        "half_page_horizontal": {COLOR_FULL: 38800,  COLOR_BW_RED: 26800,  COLOR_BW: None},
        "half_page_vertical":   {COLOR_FULL: 78000,  COLOR_BW_RED: 53800,  COLOR_BW: None},
        "third_page_vertical":  {COLOR_FULL: 50000,  COLOR_BW_RED: 35000,  COLOR_BW: None},
        "quarter_page_vertical":{COLOR_FULL: 38000,  COLOR_BW_RED: 26000,  COLOR_BW: None},
        "quarter_page_horizontal":{COLOR_FULL: 25000, COLOR_BW_RED: 18000, COLOR_BW: 12000},
        "sixth_page_vertical":  {COLOR_FULL: 31800,  COLOR_BW_RED: 21800,  COLOR_BW: 15000},
        "sixth_page_horizontal":{COLOR_FULL: 19000,  COLOR_BW_RED: 13000,  COLOR_BW: 9000},
        "eighth_page":          {COLOR_FULL: 12600,  COLOR_BW_RED: 8700,   COLOR_BW: 6100},
        "tenth_page":           {COLOR_FULL: 6400,   COLOR_BW_RED: 4500,   COLOR_BW: 3100},
        "cm_other":             {COLOR_FULL: 0,      COLOR_BW_RED: 0,      COLOR_BW: 0}
    }

    SIZE_PERCENT_TO_SPEC = [
        (95, "full_page"), (70, "small_full_page"), (55, "half_page_vertical"),
        (45, "half_page_horizontal"), (35, "third_page_vertical"),
        (25, "quarter_page_vertical"), (20, "quarter_page_horizontal"),
        (15, "sixth_page_vertical"), (12, "sixth_page_horizontal"),
        (8, "eighth_page"), (5, "tenth_page"), (0, "cm_other")
    ]

    # ==================== HELPER FUNCTIONS ====================

    def ad_size_percent_to_spec(percent: float) -> str:
        if pd.isna(percent): return "cm_other"
        try: pct_val = float(percent)
        except ValueError: return "cm_other"
        for threshold, spec in SIZE_PERCENT_TO_SPEC:
            if pct_val >= threshold: return spec
        return "cm_other"

    def normalize_color_smart(color: str, ad_type: str) -> str:
        if pd.notna(color) and str(color).strip() != "":
            c = str(color).strip().lower()
            if c in ("4c", "full_color", "full", "color", "colour", "彩色"): return COLOR_FULL
            if c in ("bw_red", "bw+red", "black_white_red", "b/w red", "套紅", "red"): return COLOR_BW_RED
            if c in ("bw", "b/w", "black_white", "monochrome", "黑白"): return COLOR_BW
            return COLOR_FULL
        
        if ad_type in ["Government", "Other Non-Commercial"]:
            return COLOR_BW
        else:
            return COLOR_FULL

    def get_ad_price(row):
        ad_type = row.get('Type', '')
        color_key = normalize_color_smart(row.get('Color', 'full_color'), ad_type)
        size_key = ad_size_percent_to_spec(row.get('Ad_Size_Percent', 0))
        
        prices = RATE_CARD_HKCD.get(size_key)
        if prices is None: return 0.0
        
        price = prices.get(color_key)
        if price is None and color_key == COLOR_BW:
            price = prices.get(COLOR_BW_RED) or prices.get(COLOR_FULL)
        if price is None and color_key == COLOR_BW_RED:
            price = prices.get(COLOR_FULL)
            
        return price if price else 0.0

    def assign_period(d):
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    def normalize_type_4class(val):
        if pd.isna(val): return None
        s = str(val).strip().lower()
        if s in ["commercial"]: return "Commercial"
        if s in ["government"]: return "Government"
        if s in ["soe"]: return "SOE"
        if s in ["other non-commercial", "other_non_commercial"]: return "Other Non-Commercial"
        return "Other Non-Commercial"

    def normalize_type_3class(val):
        if pd.isna(val): return None
        s = str(val).strip().lower()
        if s in ["government", "soe"]: return "Public Sector"
        if s in ["commercial"]: return "Commercial"
        return "Other Non-Commercial"

    # ==================== MAIN LOGIC ====================
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    hkcd = df[df['Newspaper'] == 'HKCD'].copy()
    if hkcd.empty: hkcd = df[df['Newspaper'].str.contains('HKCD|Commercial', case=False, na=False)].copy()
    if hkcd.empty: return print("No data found for HKCD.")

    hkcd['Date'] = pd.to_datetime(hkcd['Date'], errors='coerce')
    hkcd = hkcd.dropna(subset=['Date'])
    hkcd['Est_Revenue_RMB'] = hkcd.apply(get_ad_price, axis=1)

    # --- EXACT OUTPUT FORMAT REQUESTED ---
    total_ads = len(hkcd)
    unique_days = hkcd['Date'].nunique()
    total_rev = hkcd['Est_Revenue_RMB'].sum()
    avg_daily = total_rev / unique_days if unique_days else 0
    avg_per_ad = total_rev / total_ads if total_ads > 0 else 0

    print("HKCD Revenue Estimation")
    print(f"Total Ads Detected: {total_ads}")
    print(f"Unique Days with Ads: {unique_days}")
    print(f"Total Estimated Revenue: RMB ¥{total_rev:,.0f}")
    print(f"Average Daily Revenue:   RMB ¥{avg_daily:,.0f}")
    print(f"Average Revenue Per Ad:  RMB ¥{avg_per_ad:,.0f}\n")

    hkcd['Period'] = hkcd['Date'].apply(assign_period)
    plot_data = hkcd[hkcd['Period'].notna()].copy()
    plot_data['Type_4Class'] = plot_data['Type'].apply(normalize_type_4class)
    plot_data['Type_3Class'] = plot_data['Type'].apply(normalize_type_3class)

    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}
    colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_3 = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # --- PLOT 1: 4-CLASS (1 ROW) ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 6))
    target_types_4 = ['Commercial', 'Government', 'SOE', 'Other Non-Commercial']
    for i, ad_type in enumerate(target_types_4):
        ax = axes1[i]
        data = plot_data[plot_data['Type_4Class'] == ad_type]
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type); continue
        avg_by_p = data.groupby('Period')['Est_Revenue_RMB'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_4[i])
        ax.set_title(f'{ad_type} Avg Revenue (4-Class)'); ax.set_ylabel('RMB')
        for bar in bars:
            if bar.get_height() > 0: ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'¥{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('HKCD/hkcd_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: hkcd_4class_revenue.png")

    # --- PLOT 2: 3-CLASS ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    target_types_3 = ['Commercial', 'Public Sector', 'Other Non-Commercial']
    for i, ad_type in enumerate(target_types_3):
        ax = axes2[i]
        data = plot_data[plot_data['Type_3Class'] == ad_type]
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type); continue
        avg_by_p = data.groupby('Period')['Est_Revenue_RMB'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_3[i])
        ax.set_title(f'{ad_type} Avg Revenue (Combined)'); ax.set_ylabel('RMB')
        for bar in bars:
            if bar.get_height() > 0: ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'¥{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('HKCD/hkcd_3class_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: hkcd_3class_combined_revenue.png")

if __name__ == "__main__":
    analyze_hkcd_final()