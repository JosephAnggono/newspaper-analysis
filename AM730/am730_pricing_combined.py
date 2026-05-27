import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot_am730_combined():
    # ==================== CONFIGURATION ====================
    # Adjust path if running from subfolder
    INPUT_CSV = "AM730/ad_reclassified_4class.csv" 
    
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

    SIZE_PERCENT_TO_SPEC = [
        (200, SPEC_DOUBLE_FULL_PAGE), 
        (95, SPEC_FULL_PAGE), 
        (70, SPEC_SMALL_FULL_PAGE),
        (55, SPEC_HALF_PAGE_HORIZONTAL), 
        (45, SPEC_HALF_PAGE_VERTICAL),
        (38, SPEC_THIRD_PAGE_HORIZONTAL), 
        (28, SPEC_THIRD_PAGE_VERTICAL),
        (25, SPEC_QUARTER_PAGE), 
        (20, SPEC_QUARTER_PAGE_HORIZONTAL),
        (12, SPEC_SIXTH_PAGE), 
        (0, SPEC_SIXTH_PAGE),
    ]

    # ==================== HELPER FUNCTIONS ====================

    def ad_size_percent_to_spec(percent: float) -> str:
        if pd.isna(percent): return SPEC_CM_OR_OTHER
        try: pct_val = float(percent)
        except ValueError: return SPEC_CM_OR_OTHER
        for threshold, spec in SIZE_PERCENT_TO_SPEC:
            if pct_val >= threshold: return spec
        return SPEC_CM_OR_OTHER

    def normalize_color(color: str) -> str:
        if pd.isna(color): return COLOR_FULL
        c = str(color).strip().lower()
        if c in ("4c", "full_color", "full", "color", "colour"): return COLOR_FULL
        if c in ("bw_red", "bw+red", "black_white_red", "b/w red"): return COLOR_BW_RED
        if c in ("bw", "b/w", "black_white", "monochrome"): return COLOR_BW
        return COLOR_FULL

    def get_day_type(date_val) -> str:
        if pd.isna(date_val): return "mon_thu"
        dt = pd.to_datetime(date_val)
        return "fri" if dt.weekday() == 4 else "mon_thu"

    def get_ad_price(row):
        day_key = get_day_type(row['Date'])
        color_key = normalize_color(row.get('Color', 'full_color'))
        size_key = ad_size_percent_to_spec(row.get('Ad_Size_Percent', 0))
        prices = RATE_CARD_PRICES[day_key].get(size_key)
        if prices is None: return 0.0
        return prices.get(color_key, 0.0)

    def assign_period(d):
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    def normalize_and_combine_type(val):
        if pd.isna(val) or val == "": return None
        s = str(val).strip().lower()
        
        if s in ["government", "soe"]:
            return "Public Sector"
        if s in ["commercial"]:
            return "Commercial"
        if s in ["other non-commercial", "other_non_commercial"]:
            return "Other Non-Commercial"
            
        if "soe" in s or "state-owned" in s or "government" in s:
            return "Public Sector"
        if "commercial" in s and "non" not in s:
            return "Commercial"
        
        return "Other Non-Commercial"

    # ==================== MAIN LOGIC ====================

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    am730 = df[df['Newspaper'] == 'am730'].copy()
    if am730.empty:
        return print("No data found for AM730.")

    am730['Date'] = pd.to_datetime(am730['Date'], errors='coerce')
    am730 = am730.dropna(subset=['Date'])
    am730['Est_Revenue'] = am730.apply(get_ad_price, axis=1)

    # Summary Stats (Requested Format)
    total_ads = len(am730)
    unique_days = am730['Date'].nunique()
    total_rev = am730['Est_Revenue'].sum()
    
    print("AM730 Revenue Estimation (Combined Gov+SOE)")
    print(f"Total AM730 Ads Detected: {total_ads}")
    print(f"Active Days in Dataset: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue: HKD ${total_rev/unique_days:,.0f}" if unique_days else "N/A")
    print(f"Average Revenue Per Ad: HKD ${total_rev/total_ads:,.0f}\n")

    # Assign Periods & Combined Types
    am730['Period'] = am730['Date'].apply(assign_period)
    plot_data = am730[am730['Period'].notna()].copy()
    plot_data['Type_Combined'] = plot_data['Type'].apply(normalize_and_combine_type)
    
    target_types = ['Commercial', 'Public Sector', 'Other Non-Commercial']
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}

    for i, ad_type in enumerate(target_types):
        ax = axes[i]
        data = plot_data[plot_data['Type_Combined'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No Data Found for\n{ad_type}', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{ad_type} Avg Revenue')
            continue

        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors[i])
        
        ax.set_title(f'{ad_type} Avg Revenue', fontsize=14)
        ax.set_ylabel('HKD')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('am730_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved as 'am730_combined_revenue.png'")

if __name__ == "__main__":
    analyze_and_plot_am730_combined()