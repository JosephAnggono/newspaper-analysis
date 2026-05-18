import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load & clean
df = pd.read_csv("newspaper_analysis/entity_sentiment_paragraphs_one_step_vs_two_step.csv")
df = df[df['one_step_llm_sentiment_score'].notna() & df['two_step_llm_sentiment_score'].notna()].copy()

# 2. Paragraph-level robustness check
para_df = df.groupby('paragraph_text').agg({
    'one_step_llm_sentiment_score': 'mean',
    'two_step_llm_sentiment_score': 'mean'
}).reset_index()
para_r, _ = stats.pearsonr(para_df['one_step_llm_sentiment_score'], para_df['two_step_llm_sentiment_score'])

# 3. Core agreement metrics
r, _ = stats.pearsonr(df['one_step_llm_sentiment_score'], df['two_step_llm_sentiment_score'])
rho, _ = stats.spearmanr(df['one_step_llm_sentiment_score'], df['two_step_llm_sentiment_score'])
diff = df['one_step_llm_sentiment_score'] - df['two_step_llm_sentiment_score']
mae = np.abs(diff).mean()
rmse = np.sqrt((diff ** 2).mean())

# 4. Directional agreement (±0.3 neutral band)
THRESH = 0.3
df['s1'] = np.where(df['one_step_llm_sentiment_score'] < -THRESH, 'neg',
                    np.where(df['one_step_llm_sentiment_score'] > THRESH, 'pos', 'neu'))
df['s2'] = np.where(df['two_step_llm_sentiment_score'] < -THRESH, 'neg',
                    np.where(df['two_step_llm_sentiment_score'] > THRESH, 'pos', 'neu'))
sign_agg = (df['s1'] == df['s2']).mean()

# 5. Factual vs evaluative split
factual = df[df['two_step_llm_paragraph_type'] == 'factual']
evaluative = df[df['two_step_llm_paragraph_type'] == 'evaluative']
factual_mae = np.abs(factual['one_step_llm_sentiment_score'] - factual['two_step_llm_sentiment_score']).mean() if len(factual) else np.nan
evaluative_mae = np.abs(evaluative['one_step_llm_sentiment_score'] - evaluative['two_step_llm_sentiment_score']).mean() if len(evaluative) else np.nan

# 6. Occurrence impact
bins = [0, 1, 3, np.inf]
labels = ['1 mention', '2-3 mentions', '4+ mentions']
df['occ_group'] = pd.cut(df['occurrence'], bins=bins, labels=labels, right=False)
occ_corrs = {}
for g in df['occ_group'].dropna().unique():
    sub = df[df['occ_group'] == g]
    if len(sub) >= 2:
        occ_corrs[g] = stats.pearsonr(sub['one_step_llm_sentiment_score'], sub['two_step_llm_sentiment_score'])[0]

# 7. Language-stratified analysis (NEW - minimal addition)
lang_results = {}
for lang in df['language'].dropna().unique():
    sub = df[df['language'] == lang]
    if len(sub) >= 2:
        r_lang, _ = stats.pearsonr(sub['one_step_llm_sentiment_score'], sub['two_step_llm_sentiment_score'])
        mae_lang = np.abs(sub['one_step_llm_sentiment_score'] - sub['two_step_llm_sentiment_score']).mean()
        # Directional agreement for this language
        s1_l = np.where(sub['one_step_llm_sentiment_score'] < -THRESH, 'neg',
                       np.where(sub['one_step_llm_sentiment_score'] > THRESH, 'pos', 'neu'))
        s2_l = np.where(sub['two_step_llm_sentiment_score'] < -THRESH, 'neg',
                       np.where(sub['two_step_llm_sentiment_score'] > THRESH, 'pos', 'neu'))
        sign_l = (s1_l == s2_l).mean()
        lang_results[lang] = {'n': len(sub), 'r': r_lang, 'mae': mae_lang, 'sign': sign_l}

# 8. Visualizations
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# scatter: raw scores
ax = axes[0, 0]
ax.scatter(df['one_step_llm_sentiment_score'], df['two_step_llm_sentiment_score'], alpha=0.6, s=20, c='steelblue', edgecolors='white')
ax.plot([-1, 1], [-1, 1], 'r--', linewidth=1.5, label='Perfect agreement')
ax.set_xlabel('One-Step Score'); ax.set_ylabel('Two-Step Score')
ax.set_title('Entity-Level Score Comparison')
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# distribution overlap
ax = axes[0, 1]
bins_hist = np.linspace(-1, 1, 21)
ax.hist(df['one_step_llm_sentiment_score'], bins=bins_hist, alpha=0.7, label='One-Step', color='steelblue', edgecolor='white')
ax.hist(df['two_step_llm_sentiment_score'], bins=bins_hist, alpha=0.7, label='Two-Step', color='coral', edgecolor='white')
ax.set_xlabel('Sentiment Score'); ax.set_ylabel('Frequency')
ax.set_title('Score Distribution Comparison')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# confusion matrix (directional)
ax = axes[0, 2]
cm = pd.crosstab(df['s1'], df['s2'], rownames=['One-step'], colnames=['Two-step'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'}, 
            linewidths=0.5, linecolor='white', square=False)
ax.set_title('Directional Agreement (counts)', fontsize=12, pad=15)
ax.set_xlabel('Two-step'); ax.set_ylabel('One-step')
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
plt.setp(ax.get_yticklabels(), rotation=0, va='center')

# disagreement by paragraph type
ax = axes[1, 0]
if not np.isnan(factual_mae):
    ax.bar(['Factual', 'Evaluative'], [factual_mae, evaluative_mae], color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Disagreement by Paragraph Type')
    for i, v in enumerate([factual_mae, evaluative_mae]):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# agreement by mention frequency
ax = axes[1, 1]
if occ_corrs:
    occ_labels = list(occ_corrs.keys())
    occ_vals = list(occ_corrs.values())
    colors = ['forestgreen' if v > 0.7 else 'orange' for v in occ_vals]
    bars = ax.bar(occ_labels, occ_vals, color=colors, edgecolor='black')
    ax.set_ylabel('Pearson Correlation'); ax.set_title('Agreement by Entity Mention Frequency')
    ax.set_ylim(0, 1); ax.axhline(0.7, color='red', linestyle='--', alpha=0.7)
    for bar, v in zip(bars, occ_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.3f}', ha='center', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Tukey mean-difference plot
ax = axes[1, 2]
mean_s = (df['one_step_llm_sentiment_score'] + df['two_step_llm_sentiment_score']) / 2
diff_s = df['one_step_llm_sentiment_score'] - df['two_step_llm_sentiment_score']

mean_diff = diff_s.mean()
std_diff = diff_s.std()
upper_loa = mean_diff + 1.96 * std_diff
lower_loa = mean_diff - 1.96 * std_diff
n_outliers = ((diff_s > upper_loa) | (diff_s < lower_loa)).sum()

ax.scatter(mean_s, diff_s, alpha=0.6, s=20, c='steelblue', edgecolors='white')
ax.axhline(mean_diff, color='red', linestyle='--', linewidth=1.5, label=f'Mean Diff: {mean_diff:.3f}')
ax.axhline(upper_loa, color='gray', linestyle=':', linewidth=1.5, label=f'Upper LoA (+1.96σ): {upper_loa:.3f}')
ax.axhline(lower_loa, color='gray', linestyle=':', linewidth=1.5, label=f'Lower LoA (-1.96σ): {lower_loa:.3f}')
ax.set_xlabel('Mean Score')
ax.set_ylabel('Difference (One-Step - Two-Step)')
ax.set_title('Tukey Mean-Difference Analysis')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('entity_sentiment_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Summary
print("Entity Level Sentiment: One-Step LLM v.s. Two-Step LLM")
print(f"Valid pairs: {len(df)} | Unique paragraphs: {para_df.shape[0]}")
print(f"Paragraph-level r (robustness check): {para_r:.3f}")

print(f"\n[Core Metrics]")
print(f"  Pearson r: {r:.3f} | Spearman ρ: {rho:.3f}")
print(f"  MAE: {mae:.3f} | RMSE: {rmse:.3f}")
print(f"  Directional agreement (±{THRESH}): {sign_agg:.1%}")

print(f"\n[Mean-Difference]")
print(f"  Mean diff: {mean_diff:.3f} | Upper LoA: {upper_loa:.3f} | Lower LoA: {lower_loa:.3f}")
print(f"  Outliers: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

print(f"\n[Subgroup Disagreement]")
if not np.isnan(factual_mae): print(f"  Factual MAE: {factual_mae:.3f} (n={len(factual)})")
if not np.isnan(evaluative_mae): print(f"  Evaluative MAE: {evaluative_mae:.3f} (n={len(evaluative)})")

if lang_results:
    print(f"\n[By Language]")
    for lang, m in lang_results.items():
        print(f"  {lang.upper()}: r={m['r']:.3f} | MAE={m['mae']:.3f} | Sign={m['sign']:.1%} (n={m['n']})")

print(f"\n[Recommendation]")
if r > 0.75 and mae < 0.2 and para_r > 0.75:
    print("  Strong agreement across all checks. ONE-STEP is recommended (simpler, same reliability).")
elif sign_agg > 0.8:
    print("  High directional agreement. ONE-STEP preferred for efficiency.")
else:
    print("  Moderate disagreement. Manual review advised before deployment.")