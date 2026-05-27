import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_the_standard():
    # ==================== CONFIGURATION ====================
    INPUT_CSV = "ad_reclassified_4class.csv"  # Adjust path if needed
    
    # Milestones (Consistent with previous analyses)
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")
    
    # The Standard Rate Card (HK$) - Based on Image 5 (Run-of-Paper ROP)
    # Effective Jan 2026 rates used as proxy for historical estimation consistency
    
    COLOR_FULL = "full_color"
    COLOR_SPOT = "spot_color" # "Red" in card
    COLOR_BW = "bw"

    SPEC_FULL_PAGE = "full_page"           # 320x260mm → 156k / 109k / 96k
    SPEC_HALF_PAGE_VERTICAL = "half_page_vertical"   # 315x129mm → 85k / 57k / 53k
    SPEC_HALF_PAGE_HORIZONTAL = "half_page_horizontal" # 155x260mm → 85k / 57k / 53k
    SPEC_JUNIOR_PAGE = "junior_page"       # 245x200mm → 85k / 57k / 53k
    SPEC_QUARTER_PAGE = "quarter_page"     # 155x129mm → 46k / 33k / 27k
    SPEC_EIGHTH_PAGE = "eighth_page"       # 155x61mm → 29k / 23k / 17k
    SPEC_CM_OR_OTHER = "cm_other"

    RATE_CARD_TS = {
        SPEC_FULL_PAGE:          {COLOR_FULL: 156000, COLOR_SPOT: 109000, COLOR_BW: 96000},
        SPEC_HALF_PAGE_VERTICAL:   {COLOR_FULL: 85000,  COLOR_SPOT: 57000,  COLOR_BW: 53000},
        SPEC_HALF_PAGE_HORIZONTAL: {COLOR_FULL: 85000,  COLOR_SPOT: 57000,  COLOR_BW: 53000},
        SPEC_JUNIOR_PAGE:          {COLOR_FULL: 85000,  COLOR_SPOT: 57000,  COLOR_BW: 53000},
        SPEC_QUARTER_PAGE:         {COLOR_FULL: 46000,  COLOR_SPOT: 33000,  COLOR_BW: 27000},
        SPEC_EIGHTH_PAGE:          {COLOR_FULL: 29000,  COLOR_SPOT: 23000,  COLOR_BW: 17000},
        SPEC_CM_OR_OTHER:          {COLOR_FULL: 0,      COLOR_SPOT: 0,      COLOR_BW: 0}
    }

    # Map Ad_Size_Percent to Spec (Based on standard broadsheet/tabloid hybrid dimensions)
    SIZE_PERCENT_TO_SPEC = [
        (95, SPEC_FULL_PAGE),
        (70, SPEC_JUNIOR_PAGE),      # Junior page is roughly 70% area of full
        (55, SPEC_HALF_PAGE_VERTICAL),
        (45, SPEC_HALF_PAGE_HORIZONTAL),
        (25, SPEC_QUARTER_PAGE),
        (10, SPEC_EIGHTH_PAGE),
        (0, SPEC_CM_OR_OTHER)
    ]

    def ad_size_percent_to_spec(percent: float) -> str:
        if pd.isna(percent): return SPEC_CM_OR_OTHER
        try: pct_val = float(percent)
        except ValueError: return SPEC_CM_OR_OTHER
        for threshold, spec in SIZE_PERCENT_TO_SPEC:
            if pct_val >= threshold:
                return spec
        return SPEC_CM_OR_OTHER

    def normalize_color(color: str) -> str:
        if pd.isna(color): return COLOR_FULL
        c = str(color).strip().lower()
        if c in ("4c", "full_color", "full", "color", "colour"): return COLOR_FULL
        if c in ("spot_color", "red", "bw_red", "bw+red", "black_white_red", "b/w red"): return COLOR_SPOT
        if c in ("bw", "b/w", "black_white", "monochrome"): return COLOR_BW
        return COLOR_FULL

    def get_ad_price(row):
        color_key = normalize_color(row.get('Color', 'full_color'))
        size_key = ad_size_percent_to_spec(row.get('Ad_Size_Percent', 0))
        prices = RATE_CARD_TS.get(size_key)
        if prices is None: return 0.0
        price = prices.get(color_key)
        if price is None and color_key == COLOR_BW:
            price = prices.get(COLOR_SPOT) or prices.get(COLOR_FULL)
        return price if price else 0.0

    def assign_period(d):
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    # RULE-BASED CLASSIFICATION (Consistent with HD/SCMP logic)
    def classify_company_name(name: str) -> str:
        if pd.isna(name) or name == "":
            return "Other Non-Commercial"
        n = str(name).strip().lower()
        
        # Government Keywords
        government_keywords = [
            'government', 'govt', 'department', 'bureau', 'office', 'authority',
            'commission', 'council', 'ministry', 'secretariat', 'legislative',
            'executive', 'judicial', 'public service', 'civil service', 'policy unit'
        ]
        if any(kw in n for kw in government_keywords):
            return "Government"
        
        # SOE Keywords (State-Owned Enterprises & Major Institutions)
        soe_keywords = [
            'mtr', 'clp', 'hkbn', 'hktel', 
            'china mobile', 'china unicom', 'china telecom',
            'bank of china', 'icbc', 'ccb', 'abc', 'bochk',
            'hang seng bank', 'standard chartered', 'hsbc', 
            'aia', 'prudential', 'manulife',
            'new world development', 'swire', 'wharf', 'cheung kong', 'li ka shing',
            'poly property', 'country garden', 'evergrande', 'sino land', 
            'sun hung kai', 'kerri properties', 'shui on land', 'fortune land'
        ]
        if any(kw in n for kw in soe_keywords):
            return "SOE"
        
        # Commercial Keywords
        commercial_keywords = [
            'bank', 'insurance', 'retail', 'shopping', 'mall', 'hotel', 'resort', 
            'travel', 'tourism', 'airline', 'telecom', 'mobile', 'tech', 'software',
            'food', 'beverage', 'restaurant', 'fashion', 'luxury', 'auto', 'car',
            'property', 'real estate', 'developer', 'construction', 'pharma', 'healthcare',
            'education', 'university', 'college', 'training', 'consulting', 'agency'
        ]
        if any(kw in n for kw in commercial_keywords):
            return "Commercial"
        
        # Default
        return "Other Non-Commercial"

    def combine_type_3class(val: str) -> str:
        if val in ["Government", "SOE"]:
            return "Public Sector"
        return val

    # ==================== MAIN LOGIC ====================
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        return print(f"Error: File '{INPUT_CSV}' not found.")

    ts = df[df['Newspaper'] == 'The Standard'].copy()
    if ts.empty:
        # Try alternative naming if needed
        ts = df[df['Newspaper'].str.contains('Standard', case=False, na=False)].copy()
        
    if ts.empty:
        return print("No data found for The Standard.")

    ts['Date'] = pd.to_datetime(ts['Date'], errors='coerce')
    ts = ts.dropna(subset=['Date'])
    
    # Apply rule-based classification
    ts['Type_Classified'] = ts['Company_Name'].apply(classify_company_name)
    
    # Calculate Revenue
    ts['Est_Revenue_HKD'] = ts.apply(get_ad_price, axis=1)

    # Summary Stats
    total_ads = len(ts)
    unique_days = ts['Date'].nunique()
    total_rev = ts['Est_Revenue_HKD'].sum()
    
    print("The Standard Revenue Estimation (HKD)")
    print(f"Total TS Ads Detected: {total_ads}")
    print(f"Active Days in Dataset: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue: HKD ${total_rev/unique_days:,.0f}" if unique_days else "N/A")
    print(f"Average Revenue Per Ad: HKD ${total_rev/total_ads:,.0f}\n")

    # Assign Periods
    ts['Period'] = ts['Date'].apply(assign_period)
    plot_data = ts[ts['Period'].notna()].copy()

    # --- PLOT 1: 4-CLASS VIEW ---
    target_types_4 = ['Commercial', 'Government', 'SOE', 'Other Non-Commercial']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    p_labels = {'P1': 'P1\n(Jan-Jun 20)', 'P2': 'P2\n(Jun 20-Jun 21)', 'P3': 'P3\n(Jun 21-Jun 22)'}

    for i, ad_type in enumerate(target_types_4):
        ax = axes[i]
        data = plot_data[plot_data['Type_Classified'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type)
            continue
            
        avg_by_p = data.groupby('Period')['Est_Revenue_HKD'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_4[i])
        
        ax.set_title(f'{ad_type} Avg Revenue (4-Class)')
        ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('the_standard_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: the_standard_4class_revenue.png")

    # --- PLOT 2: 3-CLASS VIEW (COMBINED) ---
    plot_data['Type_Combined'] = plot_data['Type_Classified'].apply(combine_type_3class)
    target_types_3 = ['Commercial', 'Public Sector', 'Other Non-Commercial']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors_3 = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, ad_type in enumerate(target_types_3):
        ax = axes[i]
        data = plot_data[plot_data['Type_Combined'] == ad_type]
        
        if data.empty:
            ax.text(0.5, 0.5, f'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ad_type)
            continue
            
        avg_by_p = data.groupby('Period')['Est_Revenue_HKD'].mean().reindex(['P1', 'P2', 'P3']).fillna(0)
        bars = ax.bar(avg_by_p.index.map(p_labels), avg_by_p.values, color=colors_3[i])
        
        ax.set_title(f'{ad_type} Avg Revenue (Combined)')
        ax.set_ylabel('HKD')
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('the_standard_3class_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: the_standard_3class_combined_revenue.png")

if __name__ == "__main__":
    analyze_the_standard()