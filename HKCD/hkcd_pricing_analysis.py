import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_hkcd_pricing():
    # ==================== CONFIGURATION ====================
    # Adjust path if running from subfolder or different location
    INPUT_CSV = "HKCD/ad_reclassified_4class.csv" 
    
    # Milestones (Consistent with AM730/SCMP)
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")
    
    # HKCD Rate Card Constants (Based on Provided Image: Inner Pages, RMB/day)
    # Note: Image shows "Color" and "Red" (BW+Red). BW is often derived or listed separately.
    # We will map 'full_color' -> Color, 'bw_red' -> Red, 'bw' -> Black & White (if available)
    
    COLOR_FULL = "full_color"
    COLOR_BW_RED = "bw_red" 
    COLOR_BW = "bw"

    SPEC_FULL_PAGE = "full_page"           # 49x31
    SPEC_HALF_PAGE_VERTICAL = "half_page_vertical"   # 24x31 (Vertical)
    SPEC_HALF_PAGE_HORIZONTAL = "half_page_horizontal" # 24x31 (Horizontal - listed as Half-page 24x31 in image? Usually H/V differ in price if layout differs, but image lists one "Half-page 24x31" at 53800/78000. Wait, image lists "Half-page 24x31" under Inner Pages with prices 53800(Red)/78000(Color). It also lists "Miniature version 31x23". Let's stick to standard mapping.)
    
    # Mapping based on Image "Inner pages" section:
    # Full version 49x31: 110000 (Red), 160000 (Color)
    # Half-page 24x31: 53800 (Red), 78000 (Color) -> Assuming this covers both H/V if not specified, or we use Vertical as default for half page.
    # Miniature version 31x23: 65800 (Red), 81800 (Color) -> This is likely Small Full Page equivalent.
    # 12x31: 26000 (Red), 38000 (Color) -> Third Page Vertical? Or Quarter? 12 is ~1/4 of 49 width? No, 12x31 is roughly 1/4 page vertical.
    # 16x31: 35000 (Red), 50000 (Color) -> Roughly 1/3 page vertical?
    # 24x15.5: 26800 (Red), 38800 (Color) -> Half Page Horizontal?
    # 10x31: 21800 (Red), 31800 (Color)
    # 8x31: 18000 (Red), 25000 (Color)
    # 12x15.5: 13000 (Red), 19000 (Color)
    # 10x15.5: 10800 (Red), 15800 (Color)
    # 8x15.5: 8700 (Red), 12600 (Color)
    # 8x8: 4500 (Red), 6400 (Color)
    # 6x8: 2300 (Red/BW? Image says 2300 for BW, 4500 for Red, 6400 for Color? No, 6x8 row has 2300(BW), -(Red), -(Color)? Wait, looking closely at image:
    # Row 6x8: BW=2300, Red=-, Color=-. 
    # Row 8x8: BW=3100, Red=4500, Color=6400.
    # Row 8x15.5: BW=6100, Red=8700, Color=12600.
    # Row 10x15.5: BW=7500, Red=10800, Color=15800.
    # Row 10x31: BW=15000, Red=21800, Color=31800.
    # Row 12x15.5: BW=9000, Red=13000, Color=19000.
    # Row 12x31: BW=None?, Red=26000, Color=38000. (Image shows blank for BW? Or maybe it's there. Let's assume proportional if missing).
    # Row 16x15.5: BW=12000, Red=18000, Color=25000.
    # Row 16x31: BW=None?, Red=35000, Color=50000.
    # Row 24x15.5: BW=None?, Red=26800, Color=38800.
    # Row Half-page 24x31: BW=None?, Red=53800, Color=78000.
    # Row Miniature 31x23: BW=None?, Red=65800, Color=81800.
    # Row Full version 49x31: BW=None?, Red=110000, Color=160000.

    # Let's define a robust lookup based on % size to closest spec in image
    
    RATE_CARD_HKCD = {
        # Spec: { Color: Price, BW_Red: Price, BW: Price }
        "full_page":          {COLOR_FULL: 160000, COLOR_BW_RED: 110000, COLOR_BW: None},
        "small_full_page":    {COLOR_FULL: 81800,  COLOR_BW_RED: 65800,  COLOR_BW: None}, # Miniature 31x23
        "half_page_horizontal": {COLOR_FULL: 38800,  COLOR_BW_RED: 26800,  COLOR_BW: None}, # 24x15.5
        "half_page_vertical":   {COLOR_FULL: 78000,  COLOR_BW_RED: 53800,  COLOR_BW: None}, # 24x31
        "third_page_vertical":  {COLOR_FULL: 50000,  COLOR_BW_RED: 35000,  COLOR_BW: None}, # 16x31 approx
        "quarter_page_vertical":{COLOR_FULL: 38000,  COLOR_BW_RED: 26000,  COLOR_BW: None}, # 12x31 approx
        "quarter_page_horizontal":{COLOR_FULL: 25000, COLOR_BW_RED: 18000, COLOR_BW: 12000}, # 16x15.5
        "sixth_page_vertical":  {COLOR_FULL: 31800,  COLOR_BW_RED: 21800,  COLOR_BW: 15000}, # 10x31 approx? Or 12x15.5? 12x15.5 is 19k/13k. Let's use 12x15.5 for quarter-horiz.
        "sixth_page_horizontal":{COLOR_FULL: 19000,  COLOR_BW_RED: 13000,  COLOR_BW: 9000},  # 12x15.5
        "eighth_page":          {COLOR_FULL: 12600,  COLOR_BW_RED: 8700,   COLOR_BW: 6100},  # 8x15.5
        "tenth_page":           {COLOR_FULL: 6400,   COLOR_BW_RED: 4500,   COLOR_BW: 3100},  # 8x8 or 6x8? 8x8 is 6400/4500/3100.
        "cm_other":             {COLOR_FULL: 0,      COLOR_BW_RED: 0,      COLOR_BW: 0}
    }

    # Mapping Ad_Size_Percent to Specs
    # Using standard approximations since HKCD uses CM dimensions
    SIZE_PERCENT_TO_SPEC = [
        (95, "full_page"),           # 49x31
        (70, "small_full_page"),     # 31x23 (~70%)
        (55, "half_page_vertical"),  # 24x31 (~50% vert)
        (45, "half_page_horizontal"),# 24x15.5 (~40% horiz)
        (35, "third_page_vertical"), # 16x31 (~33% vert)
        (25, "quarter_page_vertical"), # 12x31 (~25% vert)
        (20, "quarter_page_horizontal"), # 16x15.5 (~20% horiz)
        (15, "sixth_page_vertical"), # 10x31 (~16% vert)
        (12, "sixth_page_horizontal"), # 12x15.5 (~12% horiz)
        (8,  "eighth_page"),         # 8x15.5
        (5,  "tenth_page"),          # 8x8
        (0,  "cm_other")
    ]

    def ad_size_percent_to_spec(percent: float) -> str:
        if pd.isna(percent): return "cm_other"
        try: pct_val = float(percent)
        except ValueError: return "cm_other"
        for threshold, spec in SIZE_PERCENT_TO_SPEC:
            if pct_val >= threshold:
                return spec
        return "cm_other"

    def normalize_color(color: str) -> str:
        if pd.isna(color): return COLOR_FULL
        c = str(color).strip().lower()
        if c in ("4c", "full_color", "full", "color", "colour", "彩色"):
            return COLOR_FULL
        if c in ("bw_red", "bw+red", "black_white_red", "b/w red", "套紅"):
            return COLOR_BW_RED
        if c in ("bw", "b/w", "black_white", "monochrome", "黑白"):
            return COLOR_BW
        return COLOR_FULL

    def get_ad_price(row):
        color_key = normalize_color(row.get('Color', 'full_color'))
        size_key = ad_size_percent_to_spec(row.get('Ad_Size_Percent', 0))
        
        prices = RATE_CARD_HKCD.get(size_key)
        if prices is None:
            return 0.0
        
        # Try exact match first
        price = prices.get(color_key)
        
        # Fallback: If BW is None, try BW_Red, then Full
        if price is None and color_key == COLOR_BW:
            price = prices.get(COLOR_BW_RED)
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

    hkcd = df[df['Newspaper'] == 'HKCD'].copy() # or 'Hong Kong Commercial Daily' depending on CSV
    if hkcd.empty:
        # Try alternative name if needed
        hkcd = df[df['Newspaper'].str.contains('HKCD|Commercial', case=False, na=False)].copy()
        
    if hkcd.empty:
        return print("No data found for HKCD.")

    hkcd['Date'] = pd.to_datetime(hkcd['Date'], errors='coerce')
    hkcd = hkcd.dropna(subset=['Date'])
    
    # Calculate Revenue
    hkcd['Est_Revenue_RMB'] = hkcd.apply(get_ad_price, axis=1)

    # Summary Stats
    total_ads = len(hkcd)
    unique_days = hkcd['Date'].nunique()
    total_rev = hkcd['Est_Revenue_RMB'].sum()
    
    print("HKCD Revenue Estimation (RMB)")
    print(f"Total HKCD Ads Detected: {total_ads}")
    print(f"Active Days in Dataset: {unique_days}")
    print(f"Total Estimated Revenue: RMB ¥{total_rev:,.0f}")
    print(f"Average Daily Revenue: RMB ¥{total_rev/unique_days:,.0f}" if unique_days else "N/A")
    print(f"Average Revenue Per Ad: RMB ¥{total_rev/total_ads:,.0f}\n")

    # Assign Periods
    hkcd['Period'] = hkcd['Date'].apply(assign_period)
    plot_data = hkcd[hkcd['Period'].notna()].copy()

    # --- PLOT 1: 4-CLASS VIEW ---
    plot_data['Type_4Class'] = plot_data['Type'].apply(normalize_type_4class)
    # If column is 'Type', change above line to: plot_data['Type_4Class'] = plot_data['Type'].apply(normalize_type_4class)
    
    target_types_4 = ['Commercial', 'Government', 'SOE', 'Other Non-Commercial']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}

    for i, ad_type in enumerate(target_types_4):
        ax = axes[i]
        data = plot_data[plot_data['Type_4Class'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type)
            continue
            
        avg_by_p = data.groupby('Period')['Est_Revenue_RMB'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_4[i])
        
        ax.set_title(f'{ad_type} Avg Revenue (4-Class)')
        ax.set_ylabel('RMB')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'¥{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('hkcd_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: hkcd_4class_revenue.png")

    # --- PLOT 2: 3-CLASS VIEW (COMBINED) ---
    plot_data['Type_3Class'] = plot_data['Type'].apply(normalize_type_3class)
    
    target_types_3 = ['Commercial', 'Public Sector', 'Other Non-Commercial']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors_3 = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, ad_type in enumerate(target_types_3):
        ax = axes[i]
        data = plot_data[plot_data['Type_3Class'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type)
            continue
            
        avg_by_p = data.groupby('Period')['Est_Revenue_RMB'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_3[i])
        
        ax.set_title(f'{ad_type} Avg Revenue (Combined)')
        ax.set_ylabel('RMB')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'¥{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('hkcd_3class_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: hkcd_3class_combined_revenue.png")

if __name__ == "__main__":
    analyze_hkcd_pricing()