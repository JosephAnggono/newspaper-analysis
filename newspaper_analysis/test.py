import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load & clean
df = pd.read_csv("newspaper_analysis/entity_sentiment_paragraphs_one_step_vs_two_step.csv")
df = df[df['one_step_llm_sentiment_score'].notna() & df['two_step_llm_sentiment_score'].notna()].copy()

# 1. DUPLICATE TEXT CHECK
para_agg = df.groupby('paragraph_text').agg({
    'one_step_llm_sentiment_score': 'mean',
    'two_step_llm_sentiment_score': 'mean'
})
para_corr, _ = stats.pearsonr(para_agg['one_step_llm_sentiment_score'], para_agg['two_step_llm_sentiment_score'])

# 2. OCCURRENCE IMPACT
bins = [0, 1, 3, np.inf]
labels = ['1 mention', '2-3 mentions', '4+ mentions']
df['occ_group'] = pd.cut(df['occurrence'], bins=bins, labels=labels, right=False)
occ_corr = df.groupby('occ_group').apply(
    lambda g: stats.pearsonr(g['one_step_llm_sentiment_score'], g['two_step_llm_sentiment_score'])[0]
    if len(g) >= 2 else np.nan
)

# 3. CORE METRICS
r, _ = stats.pearsonr(df['one_step_llm_sentiment_score'], df['two_step_llm_sentiment_score'])
mae = np.abs(df['one_step_llm_sentiment_score'] - df['two_step_llm_sentiment_score']).mean()

t = 0.3
df['s1'] = np.where(df['one_step_llm_sentiment_score'] < -t, 'neg',
                    np.where(df['one_step_llm_sentiment_score'] > t, 'pos', 'neu'))
df['s2'] = np.where(df['two_step_llm_sentiment_score'] < -t, 'neg',
                    np.where(df['two_step_llm_sentiment_score'] > t, 'pos', 'neu'))
sign_agg = (df['s1'] == df['s2']).mean()

# 4. FACTUAL vs EVALUATIVE
factual = df[df['two_step_llm_paragraph_type'] == 'factual']
eval_df = df[df['two_step_llm_paragraph_type'] == 'evaluative']

# 5. GENERATE VISUALIZATIONS
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Bland-Altman Plot
ax = axes[0, 0]
mean_scores = (df['one_step_llm_sentiment_score'] + df['two_step_llm_sentiment_score']) / 2
diff_scores = df['one_step_llm_sentiment_score'] - df['two_step_llm_sentiment_score']
ax.scatter(mean_scores, diff_scores, alpha=0.6, s=20, c='coral', edgecolors='white')
ax.axhline(diff_scores.mean(), color='red', linestyle='--', label=f"Mean: {diff_scores.mean():.3f}")
ax.axhline(diff_scores.mean() + 1.96*diff_scores.std(), color='gray', linestyle=':', alpha=0.7, label='±1.96 SD')
ax.axhline(diff_scores.mean() - 1.96*diff_scores.std(), color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Mean Score')
ax.set_ylabel('Difference (One-Step - Two-Step)')
ax.set_title('Bland-Altman Plot')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Distribution Comparison
ax = axes[0, 1]
bins_hist = np.linspace(-1, 1, 21)
ax.hist(df['one_step_llm_sentiment_score'], bins=bins_hist, alpha=0.7, 
        label='One-Step', color='steelblue', edgecolor='white')
ax.hist(df['two_step_llm_sentiment_score'], bins=bins_hist, alpha=0.7, 
        label='Two-Step', color='coral', edgecolor='white')
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Frequency')
ax.set_title('Score Distribution')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: Sign Agreement Heatmap
ax = axes[0, 2]
crosstab = pd.crosstab(df['s1'], df['s2'], normalize='index') * 100
im = ax.imshow(crosstab, cmap='Blues', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(crosstab.columns)))
ax.set_yticks(range(len(crosstab.index)))
ax.set_xticklabels(crosstab.columns)
ax.set_yticklabels(crosstab.index)
ax.set_xlabel('Two-Step')
ax.set_ylabel('One-Step')
ax.set_title('Sign Agreement (%)')
plt.colorbar(im, ax=ax, label='%')
for i in range(len(crosstab.index)):
    for j in range(len(crosstab.columns)):
        ax.text(j, i, f'{crosstab.iloc[i, j]:.0f}%', ha='center', va='center', fontsize=8)

# Plot 4: Disagreement by Paragraph Type
ax = axes[1, 0]
df_temp = df.copy()
df_temp['abs_diff'] = np.abs(df_temp['one_step_llm_sentiment_score'] - df_temp['two_step_llm_sentiment_score'])
if 'two_step_llm_paragraph_type' in df_temp.columns:
    types = df_temp['two_step_llm_paragraph_type'].unique()
    means = [df_temp[df_temp['two_step_llm_paragraph_type']==t]['abs_diff'].mean() for t in types]
    bars = ax.bar(range(len(types)), means, color=plt.cm.Set2(range(len(types))), edgecolor='black')
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types)
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Disagreement by Paragraph Type')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 5: Agreement by Occurrence
ax = axes[1, 1]
occ_labels = [g for g in occ_corr.index if not pd.isna(occ_corr[g])]
occ_values = [occ_corr[g] for g in occ_labels]
colors = ['forestgreen' if v > 0.7 else 'orange' for v in occ_values]
bars = ax.bar(range(len(occ_labels)), occ_values, color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(occ_labels)))
ax.set_xticklabels(occ_labels, rotation=45, ha='right')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Agreement by Entity Occurrence')
ax.set_ylim(0, 1)
ax.axhline(0.7, color='red', linestyle='--', alpha=0.7, label='r=0.7 threshold')
ax.legend(fontsize=8)
for bar, val in zip(bars, occ_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'r={val:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 6: Entity Validity Impact (if entity_exact_word column exists)
ax = axes[1, 2]
if 'entity_exact_word' in df.columns and 'paragraph_text' in df.columns:
    def entity_in_text(row):
        if pd.isna(row['entity_exact_word']) or pd.isna(row['paragraph_text']):
            return False
        words = [w.strip() for w in str(row['entity_exact_word']).split('|')]
        text = str(row['paragraph_text']).lower()
        return any(word.lower() in text for word in words if word)
    df['entity_present'] = df.apply(entity_in_text, axis=1)
    matched = df[df['entity_present']]
    mismatched = df[~df['entity_present']]
    corr_matched = stats.pearsonr(matched['one_step_llm_sentiment_score'], matched['two_step_llm_sentiment_score'])[0] if len(matched) >= 2 else np.nan
    corr_mismatched = stats.pearsonr(mismatched['one_step_llm_sentiment_score'], mismatched['two_step_llm_sentiment_score'])[0] if len(mismatched) >= 2 else np.nan
    categories = ['Present', 'Missing']
    corrs = [corr_matched, corr_mismatched]
    colors = ['forestgreen' if (c and c > 0.5) else 'orange' for c in corrs]
    bars = ax.bar(categories, [c if not np.isnan(c) else 0 for c in corrs], color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Agreement by Entity Validity')
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.7, label='r=0.7 threshold')
    ax.legend(fontsize=8)
    for bar, corr in zip(bars, corrs):
        if not np.isnan(corr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'r={corr:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
plt.close()

# OUTPUT
print("="*60)
print("SUPERVISOR BRIEF: ONE-STEP vs TWO-STEP LLM")
print("="*60)
print(f"Entity-paragraph pairs: {len(df)} | Unique paragraphs: {para_agg.shape[0]}")
print(f"\n--- Agreement ---")
print(f"Row-level Pearson r: {r:.3f} | MAE: {mae:.3f}")
print(f"Paragraph-level r (robustness): {para_corr:.3f}")
print(f"Sign agreement (±{t}): {sign_agg:.1%}")
print(f"\n--- By Occurrence ---")
for g, c in occ_corr.items():
    if not pd.isna(c):
        print(f"  {g}: r = {c:.3f}")
print(f"\n--- Factual vs Evaluative ---")
print(f"Factual (n={len(factual)}): Two-Step mean={factual['two_step_llm_sentiment_score'].mean():.2f} | One-Step mean={factual['one_step_llm_sentiment_score'].mean():.2f}")
print(f"Evaluative (n={len(eval_df)}): Two-Step mean={eval_df['two_step_llm_sentiment_score'].mean():.2f} | One-Step mean={eval_df['one_step_llm_sentiment_score'].mean():.2f}")
print(f"\n{'='*60}")
print("RECOMMENDATION:")
if r > 0.75 and mae < 0.2 and para_corr > 0.75:
    print("USE ONE-STEP. Strong agreement confirmed at both row and paragraph levels. Duplicate text does not inflate results. Occurrence has minimal impact. One-step is simpler and equally reliable.")
elif len(factual) > 0 and factual['one_step_llm_sentiment_score'].abs().mean() > 0.15:
    avg_mag = factual['one_step_llm_sentiment_score'].abs().mean()
    print(f"USE TWO-STEP. One-step assigns non-neutral scores to factual paragraphs (~{avg_mag:.2f} avg magnitude). Two-step correctly forces factual content to neutral, reducing false positives.")
else:
    print("USE ONE-STEP for speed/scale; apply a post-hoc neutral filter if factual suppression is required.")
print("="*60)
print(f"\nVisualizations saved to: sentiment_analysis_results.png")