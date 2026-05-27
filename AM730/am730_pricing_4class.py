import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot_am730_4class():
    # ==================== CONFIGURATION ====================
    INPUT_CSV = "ad_reclassified_4class.csv" # MUST use the 4-class file
    
    # Milestones for Periods (Consistent with SCMP Task)
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")
    
    # Rate Card Constants (From AM730 Ratecard No. 20, Jan 2025)
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

    SIZE_OPTIONS = [
        SPEC_FULL_PAGE, SPEC_DOUBLE_FULL_PAGE, SPEC_HALF_PAGE_VERTICAL,
        SPEC_HALF_PAGE_HORIZONTAL, SPEC_SMALL_FULL_PAGE, SPEC_THIRD_PAGE_HORIZONTAL,
        SPEC_THIRD_PAGE_VERTICAL, SPEC_QUARTER_PAGE, SPEC_QUARTER_PAGE_HORIZONTAL,
        SPEC_SIXTH_PAGE, SPEC_CM_OR_OTHER,
    ]

    # AM730 ROP Rate Card (HK$) - Extracted from PDF
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

    # Mapping Ad_Size_Percent to Rate Card Specs
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
        (0, SPEC_SIXTH_PAGE), # Default fallback
    ]

    # ==================== HELPER FUNCTIONS ====================

    def ad_size_percent_to_spec(percent: float) -> str:
        """Map Ad_Size_Percent to the largest spec it fits into."""
        if pd.isna(percent):
            return SPEC_CM_OR_OTHER
        try:
            pct_val = float(percent)
        except ValueError:
            return SPEC_CM_OR_OTHER
            
        for threshold, spec in SIZE_PERCENT_TO_SPEC:
            if pct_val >= threshold:
                return spec
        return SPEC_CM_OR_OTHER

    def normalize_color(color: str) -> str:
        """Normalize color string to rate card keys."""
        if pd.isna(color):
            return COLOR_FULL # Default assumption
        c = str(color).strip().lower()
        if c in ("4c", "full_color", "full", "color", "colour"):
            return COLOR_FULL
        if c in ("bw_red", "bw+red", "black_white_red", "b/w red"):
            return COLOR_BW_RED
        if c in ("bw", "b/w", "black_white", "monochrome"):
            return COLOR_BW
        return COLOR_FULL

    def get_day_type(date_val) -> str:
        """Returns 'fri' or 'mon_thu' based on date."""
        if pd.isna(date_val):
            return "mon_thu"
        dt = pd.to_datetime(date_val)
        if dt.weekday() == 4: # Friday
            return "fri"
        return "mon_thu"

    def get_ad_price(row):
        """Calculate price based on Date, Color, and Size Percent."""
        day_key = get_day_type(row['Date'])
        color_key = normalize_color(row.get('Color', 'full_color'))
        
        # Map Percent to Spec
        size_key = ad_size_percent_to_spec(row.get('Ad_Size_Percent', 0))
        
        prices = RATE_CARD_PRICES[day_key].get(size_key)
        if prices is None:
            return 0.0
        return prices.get(color_key, 0.0)

    def assign_period(d):
        """Assign P1, P2, P3 based on SCMP-consistent dates."""
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    def normalize_classification(val):
        """
        Map Type to 4 Classes: Commercial, Government, SOE, Other Non-Commercial.
        Since we are using ad_reclassified_4class.csv, the 'Type' column should already be clean.
        We just normalize casing/spaces.
        """
        if pd.isna(val) or val == "":
            return None
        s = str(val).strip().lower()
        
        # Handle common variations
        if s in ["commercial"]:
            return "Commercial"
        if s in ["government"]:
            return "Government"
        if s in ["soe"]:
            return "SOE"
        if s in ["other non-commercial", "other_non_commercial"]:
            return "Other Non-Commercial"
            
        # Fallback logic if data is messy
        if "soe" in s or "state-owned" in s:
            return "SOE"
        if "commercial" in s and "non" not in s:
            return "Commercial"
        if "government" in s:
            return "Government"
        
        return "Other Non-Commercial"

    # ==================== MAIN LOGIC ====================

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    # 2. Filter for AM730
    am730 = df[df['Newspaper'] == 'am730'].copy()
    
    if am730.empty:
        return print("No data found for AM730.")

    # 3. Preprocess Dates
    am730['Date'] = pd.to_datetime(am730['Date'], errors='coerce')
    am730 = am730.dropna(subset=['Date'])

    # 4. Calculate Price
    am730['Est_Revenue'] = am730.apply(get_ad_price, axis=1)

    # 5. Summary Stats
    total_ads = len(am730)
    unique_days = am730['Date'].nunique()
    total_rev = am730['Est_Revenue'].sum()
    
    avg_per_ad = total_rev / total_ads if total_ads > 0 else 0
    avg_daily = total_rev / unique_days if unique_days else 0

    print("AM730 Revenue Estimation")
    print(f"Total Ads Detected: {total_ads}")
    print(f"Unique Days with Ads: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue:   HKD ${avg_daily:,.0f}")
    print(f"Average Revenue Per Ad:  HKD ${avg_per_ad:,.0f}\n")

    # 6. Assign Periods & Classifications
    am730['Period'] = am730['Date'].apply(assign_period)
    plot_data = am730[am730['Period'].notna()].copy()
    
    # Use the existing 'Type' column from the 4-class file
    plot_data['Type_Clean'] = plot_data['Type'].apply(normalize_classification)
    
    # Define target types for 4-class analysis
    target_types = ['Commercial', 'Government', 'SOE', 'Other Non-Commercial']
    
    # 7. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}

    for i, ad_type in enumerate(target_types):
        ax = axes[i]
        
        # Filter data for this type
        data = plot_data[plot_data['Type_Clean'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No Data Found for\n{ad_type}', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{ad_type} Avg Revenue')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Calculate Average Revenue per Period
        avg_by_p = data.groupby('Period')['Est_Revenue'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors[:3]) # Use first 3 colors for consistency
        
        ax.set_title(f'{ad_type} Avg Revenue', fontsize=14)
        ax.set_ylabel('HKD')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('am730_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved as 'am730_4class_revenue.png'")

if __name__ == "__main__":
    analyze_and_plot_am730_4class()