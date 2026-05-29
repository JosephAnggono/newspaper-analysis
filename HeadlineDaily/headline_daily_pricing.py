import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_headline_daily():
    # ==================== CONFIGURATION ====================
    INPUT_CSV = "Datasets/ad_reclassified_4class.csv"  # Adjust path if needed
    
    # Milestones (Consistent with AM730/SCMP/HKCD)
    MILESTONE_P1_START = pd.Timestamp("2020-01-01")
    MILESTONE_P1_END = pd.Timestamp("2020-06-28")
    MILESTONE_P2_START = pd.Timestamp("2020-06-28")
    MILESTONE_P2_END = pd.Timestamp("2021-06-13")
    MILESTONE_P3_START = pd.Timestamp("2021-06-17")
    MILESTONE_P3_END = pd.Timestamp("2022-06-17")
    
    # Headline Daily Rate Card (HK$) - Based on Provided Image (Effective Jan 1, 2026)
    # Run-of-Page (ROP) Inner Pages Only (Special Positions ignored for simplicity unless specified)
    
    COLOR_FULL = "full_color"
    COLOR_BW_RED = "bw_red"
    COLOR_BW = "bw"

    SPEC_FULL_PAGE = "full_page"           # 265x327mm → 460,000 / 331,000 / 285,000
    SPEC_HALF_PAGE_VERTICAL = "half_page_vertical"   # 130x327mm → 241,000 / 175,000 / 150,000
    SPEC_HALF_PAGE_HORIZONTAL = "half_page_horizontal" # 265x160mm → 241,000 / 175,000 / 150,000
    SPEC_QUARTER_PAGE = "quarter_page"     # 130x160mm → 120,000 / 88,000 / 75,000
    SPEC_SIXTH_PAGE = "sixth_page"         # 85x160mm → 85,000 / 59,000 / 54,000
    SPEC_JUNIOR_PAGE = "junior_page"       # 175x215mm → 294,000 / 209,000 / 181,000
    SPEC_CM_OR_OTHER = "cm_other"

    RATE_CARD_HD = {
        SPEC_FULL_PAGE:          {COLOR_FULL: 460000, COLOR_BW_RED: 331000, COLOR_BW: 285000},
        SPEC_HALF_PAGE_VERTICAL:   {COLOR_FULL: 241000, COLOR_BW_RED: 175000, COLOR_BW: 150000},
        SPEC_HALF_PAGE_HORIZONTAL: {COLOR_FULL: 241000, COLOR_BW_RED: 175000, COLOR_BW: 150000},
        SPEC_QUARTER_PAGE:         {COLOR_FULL: 120000, COLOR_BW_RED: 88000,  COLOR_BW: 75000},
        SPEC_SIXTH_PAGE:           {COLOR_FULL: 85000,  COLOR_BW_RED: 59000,  COLOR_BW: 54000},
        SPEC_JUNIOR_PAGE:          {COLOR_FULL: 294000, COLOR_BW_RED: 209000, COLOR_BW: 181000},
        SPEC_CM_OR_OTHER:          {COLOR_FULL: 0,      COLOR_BW_RED: 0,      COLOR_BW: 0}
    }

    # Map Ad_Size_Percent to Spec (Approximate based on mm dimensions vs % of full page)
    SIZE_PERCENT_TO_SPEC = [
        (95, SPEC_FULL_PAGE),
        (70, SPEC_JUNIOR_PAGE),      # 175x215 ≈ 70% of full page area? Approximation.
        (55, SPEC_HALF_PAGE_VERTICAL),
        (45, SPEC_HALF_PAGE_HORIZONTAL),
        (25, SPEC_QUARTER_PAGE),
        (15, SPEC_SIXTH_PAGE),
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
        if c in ("bw_red", "bw+red", "black_white_red", "b/w red"): return COLOR_BW_RED
        if c in ("bw", "b/w", "black_white", "monochrome"): return COLOR_BW
        return COLOR_FULL

    def get_ad_price(row):
        color_key = normalize_color(row.get('Color', 'full_color'))
        size_key = ad_size_percent_to_spec(row.get('Ad_Size_Percent', 0))
        prices = RATE_CARD_HD.get(size_key)
        if prices is None: return 0.0
        price = prices.get(color_key)
        if price is None and color_key == COLOR_BW:
            price = prices.get(COLOR_BW_RED) or prices.get(COLOR_FULL)
        return price if price else 0.0

    def assign_period(d):
        if MILESTONE_P1_START <= d < MILESTONE_P1_END: return 'P1'
        if MILESTONE_P2_START <= d < MILESTONE_P2_END: return 'P2'
        if MILESTONE_P3_START <= d < MILESTONE_P3_END: return 'P3'
        return None

    # RULE-BASED CLASSIFICATION (Created from scratch, similar to SCMP)
    def classify_company_name(name: str) -> str:
        if pd.isna(name) or name == "":
            return "Other Non-Commercial"
        n = str(name).strip().lower()
        
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
        
        # Government Keywords
        government_keywords = [
            'government', 'govt', 'department', 'bureau', 'office', 'authority',
            'commission', 'council', 'ministry', 'secretariat', 'legislative',
            'executive', 'judicial', 'public service', 'civil service', 'policy unit'
        ]
        if any(kw in n for kw in government_keywords):
            return "Government"
        
        # SOE Keywords (State-Owned Enterprises)
        soe_keywords = [
            'mtr', 'clp', 'hkbn', 'hktel', 'china mobile', 'china unicom', 'china telecom',
            'bank of china', 'icbc', 'ccb', 'abc', 'bochk', 'hang seng bank', 'standard chartered',
            'hsbc', 'aia', 'prudential', 'manulife', 'new world development', 'swire', 'wharf',
            'cheung kong', 'li ka shing', 'poly property', 'country garden', 'evergrande',
            'sino land', 'sun hung kai', 'kerri properties', 'shui on land', 'fortune land'
        ]
        if any(kw in n for kw in soe_keywords):
            return "SOE"
        
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

    hd = df[df['Newspaper'] == 'HeadlineDaily'].copy()
    if hd.empty:
        return print("No data found for Headline Daily.")

    hd['Date'] = pd.to_datetime(hd['Date'], errors='coerce')
    hd = hd.dropna(subset=['Date'])
    
    # Apply rule-based classification
    hd['Type_Classified'] = hd['Company_Name'].apply(classify_company_name)
    
    # Calculate Revenue
    hd['Est_Revenue_HKD'] = hd.apply(get_ad_price, axis=1)

    # Summary Stats
    total_ads = len(hd)
    unique_days = hd['Date'].nunique()
    total_rev = hd['Est_Revenue_HKD'].sum()
    
    print("Headline Daily Revenue Estimation (HKD)")
    print(f"Total HD Ads Detected: {total_ads}")
    print(f"Active Days in Dataset: {unique_days}")
    print(f"Total Estimated Revenue: HKD ${total_rev:,.0f}")
    print(f"Average Daily Revenue: HKD ${total_rev/unique_days:,.0f}" if unique_days else "N/A")
    print(f"Average Revenue Per Ad: HKD ${total_rev/total_ads:,.0f}\n")

    # Assign Periods
    hd['Period'] = hd['Date'].apply(assign_period)
    plot_data = hd[hd['Period'].notna()].copy()

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
    plt.savefig('HeadlineDaily/headline_daily_4class_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: HeadlineDaily/headline_daily_4class_revenue.png")

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
    plt.savefig('HeadlineDaily/headline_daily_3class_combined_revenue.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: HeadlineDaily/headline_daily_3class_combined_revenue.png")

if __name__ == "__main__":
    analyze_headline_daily()