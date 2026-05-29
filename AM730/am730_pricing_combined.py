import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_am730_final():
    # ==================== CONFIGURATION ====================
    INPUT_CSV = "Datasets/ad_reclassified_4class.csv" 
    
    # Milestones
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")
    
    # Rate Card Constants
    COLOR_FULL = "full_color"
    COLOR_BW_RED = "bw_red"
    COLOR_BW = "bw"

    SPEC_FULL_PAGE = "full_page"
    SPEC_DOUBLE_FULL_PAGE = "double_full_page"
    SPEC_HALF_PAGE_VERTICAL = "half_page_vertical"
    SPEC_HALF_PAGE_HORIZONTAL = "half_page_horizontal"
    SPEC_SMALL_FULL_PAGE = "small_full_page"
    SPEC_THIRD_PAGE_HORIZONTAL = "one_third_page_horizontal"
    SPEC_THIRD_PAGE_VERTICAL = "one_third_page_vertical"
    SPEC_QUARTER_PAGE = "quarter_page"
    SPEC_QUARTER_PAGE_HORIZONTAL = "quarter_page_horizontal"
    SPEC_SIXTH_PAGE = "sixth_page"
    SPEC_CM_OR_OTHER = "cm_column_or_other"

    # AM730 ROP Rate Card (HK$)
    RATE_CARD_PRICES = {
        "mon_thu": {
            SPEC_FULL_PAGE: {COLOR_FULL: 284000, COLOR_BW_RED: 253000, COLOR_BW: 234000},
            SPEC_DOUBLE_FULL_PAGE: {COLOR_FULL: 624800, COLOR_BW_RED: 556600, COLOR_BW: 514800},
            SPEC_HALF_PAGE_VERTICAL: {COLOR_FULL: 156000, COLOR_BW_RED: 142000, COLOR_BW: 130000},
            SPEC_HALF_PAGE_HORIZONTAL: {COLOR_FULL: 170000, COLOR_BW_RED: 149000, COLOR_BW: 135000},
            SPEC_SMALL_FULL_PAGE: {COLOR_FULL: 184000, COLOR_BW_RED: 165000, COLOR_BW: 154000},
            SPEC_THIRD_PAGE_HORIZONTAL: {COLOR_FULL: 150000, COLOR_BW_RED: 133000, COLOR_BW: 123000},
            SPEC_THIRD_PAGE_VERTICAL: {COLOR_FULL: 145000, COLOR_BW_RED: 130000, COLOR_BW: 119000},
            SPEC_QUARTER_PAGE: {COLOR_FULL: 86000, COLOR_BW_RED: 78000, COLOR_BW: 69000},
            SPEC_QUARTER_PAGE_HORIZONTAL: {COLOR_FULL: 95000, COLOR_BW_RED: 84000, COLOR_BW: 78000},
            SPEC_SIXTH_PAGE: {COLOR_FULL: 58000, COLOR_BW_RED: 53000, COLOR_BW: 47000},
            SPEC_CM_OR_OTHER: None,
        },
        "fri": {
            SPEC_FULL_PAGE: {COLOR_FULL: 311000, COLOR_BW_RED: 278000, COLOR_BW: 257000},
            SPEC_DOUBLE_FULL_PAGE: {COLOR_FULL: 684200, COLOR_BW_RED: 611600, COLOR_BW: 565400},
            SPEC_HALF_PAGE_VERTICAL: {COLOR_FULL: 172000, COLOR_BW_RED: 156000, COLOR_BW: 144000},
            SPEC_HALF_PAGE_HORIZONTAL: {COLOR_FULL: 187000, COLOR_BW_RED: 164000, COLOR_BW: 149000},
            SPEC_SMALL_FULL_PAGE: {COLOR_FULL: 203000, COLOR_BW_RED: 182000, COLOR_BW: 170000},
            SPEC_THIRD_PAGE_HORIZONTAL: {COLOR_FULL: 165000, COLOR_BW_RED: 147000, COLOR_BW: 134000},
            SPEC_THIRD_PAGE_VERTICAL: {COLOR_FULL: 160000, COLOR_BW_RED: 144000, COLOR_BW: 131000},
            SPEC_QUARTER_PAGE: {COLOR_FULL: 96000, COLOR_BW_RED: 85000, COLOR_BW: 77000},
            SPEC_QUARTER_PAGE_HORIZONTAL: {COLOR_FULL: 103000, COLOR_BW_RED: 93000, COLOR_BW: 85000},
            SPEC_SIXTH_PAGE: {COLOR_FULL: 63000, COLOR_BW_RED: 59000, COLOR_BW: 51000},
            SPEC_CM_OR_OTHER: None,
        },
    }

    # ==================== HELPER FUNCTIONS ====================

    def ad_size_percent_to_spec_improved(percent: float) -> str:
        if pd.isna(percent): return SPEC_CM_OR_OTHER
        try: pct_val = float(percent)
        except ValueError: return SPEC_CM_OR_OTHER
        
        # IMPROVEMENT: Map >=150% to Double Full Page
        if pct_val >= 150: return SPEC_DOUBLE_FULL_PAGE
        elif pct_val >= 95: return SPEC_FULL_PAGE
        elif pct_val >= 70: return SPEC_SMALL_FULL_PAGE
        elif pct_val >= 55: return SPEC_HALF_PAGE_HORIZONTAL
        elif pct_val >= 45: return SPEC_HALF_PAGE_VERTICAL
        elif pct_val >= 38: return SPEC_THIRD_PAGE_HORIZONTAL
        elif pct_val >= 28: return SPEC_THIRD_PAGE_VERTICAL
        elif pct_val >= 25: return SPEC_QUARTER_PAGE
        elif pct_val >= 20: return SPEC_QUARTER_PAGE_HORIZONTAL
        elif pct_val >= 12: return SPEC_SIXTH_PAGE
        else: return SPEC_CM_OR_OTHER

    def normalize_color_smart(color: str, ad_type: str) -> str:
        # IMPROVEMENT: Default Gov/Non-Commercial to B/W if missing
        if pd.notna(color) and str(color).strip() != "":
            c = str(color).strip().lower()
            if c in ("4c", "full_color", "full", "color", "colour"): return COLOR_FULL
            if c in ("bw_red", "bw+red", "black_white_red", "b/w red", "red"): return COLOR_BW_RED
            if c in ("bw", "b/w", "black_white", "monochrome"): return COLOR_BW
            return COLOR_FULL
        
        if ad_type in ["Government", "Other Non-Commercial"]:
            return COLOR_BW
        else:
            return COLOR_FULL

    def get_day_type(date_val) -> str:
        if pd.isna(date_val): return "mon_thu"
        dt = pd.to_datetime(date_val)
        return "fri" if dt.weekday() == 4 else "mon_thu"

    def get_ad_price(row):
        day_key = get_day_type(row['Date'])
        ad_type = row.get('Type', '') 
        color_key = normalize_color_smart(row.get('Color', ''), ad_type)
        size_key = ad_size_percent_to_spec_improved(row.get('Ad_Size_Percent', 0))
        
        prices = RATE_CARD_PRICES[day_key].get(size_key)
        if prices is None: return 0.0
        return prices.get(color_key, 0.0)

    def assign_period(d):
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    def normalize_classification(val):
        if pd.isna(val) or val == "": return None
        s = str(val).strip().lower()
        if s in ["commercial"]: return "Commercial"
        if s in ["government"]: return "Government"
        if s in ["soe"]: return "SOE"
        if s in ["other non-commercial", "other_non_commercial"]: return "Other Non-Commercial"
        if "soe" in s or "state-owned" in s: return "SOE"
        if "commercial" in s and "non" not in s: return "Commercial"
        if "government" in s: return "Government"
        return "Other Non-Commercial"

    def combine_type(val):
        if val in ["Government", "SOE"]: return "Public Sector"
        return val

    # ==================== MAIN LOGIC ====================
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    am730 = df[df['Newspaper'] == 'am730'].copy()
    if am730.empty: return print("No data found for AM730.")

    am730['Date'] = pd.to_datetime(am730['Date'], errors='coerce')
    am730 = am730.dropna(subset=['Date'])
    am730['Est_Revenue'] = am730.apply(get_ad_price, axis=1)

    # --- EXACT OUTPUT FORMAT REQUESTED ---
    total_ads = len(am730)
    unique_days = am730['Date'].nunique()
    total_rev = am730['Est_Revenue'].sum()
    avg_daily = total_rev / unique_days if unique_days else 0
    avg_per_ad = total_rev / total_ads if total_ads > 0 else 0

    print("AM730 Revenue Estimation")
    print(f"Total Ads Detected: {total_ads}")
    print(f"Unique Days with Ads: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue:   HKD ${avg_daily:,.0f}")
    print(f"Average Revenue Per Ad:  HKD ${avg_per_ad:,.0f}\n")

    am730['Period'] = am730['Date'].apply(assign_period)
    plot_data = am730[am730['Period'].notna()].copy()
    plot_data['Type_Clean'] = plot_data['Type'].apply(normalize_classification)
    plot_data['Type_Combined'] = plot_data['Type_Clean'].apply(combine_type)

    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}
    colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_3 = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # --- PLOT 1: 4-CLASS (1 ROW) ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 6))
    target_types_4 = ['Commercial', 'Government', 'SOE', 'Other Non-Commercial']

    for i, ad_type in enumerate(target_types_4):
        ax = axes1[i]
        data = plot_data[plot_data['Type_Clean'] == ad_type]
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type); ax.set_xticks([]); ax.set_yticks([])
            continue
        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_4[i])
        ax.set_title(f'{ad_type} Avg Revenue (4-Class)'); ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0: ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('AM730/am730_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: am730_4class_revenue.png")

    # --- PLOT 2: 3-CLASS (COMBINED) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    target_types_3 = ['Commercial', 'Public Sector', 'Other Non-Commercial']

    for i, ad_type in enumerate(target_types_3):
        ax = axes2[i]
        data = plot_data[plot_data['Type_Combined'] == ad_type]
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type); ax.set_xticks([]); ax.set_yticks([])
            continue
        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_3[i])
        ax.set_title(f'{ad_type} Avg Revenue (Combined)'); ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0: ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('AM730/am730_3class_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: am730_3class_combined_revenue.png")

if __name__ == "__main__":
    analyze_am730_final()