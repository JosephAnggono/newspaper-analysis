import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

def load_data():
    file_path = 'newspaper_analysis/copy_of_entity_sentiment_paragraphs_one_step_vs_two_step.csv'
    df = pd.read_csv(file_path)
    
    # Keep only rows with human labels and model scores
    df = df.dropna(subset=['human_label_sentiment', 'one_step_llm_sentiment_score', 'two_step_llm_sentiment_score'])
    return df

def preprocess_labels(df):
    """
    Maps human labels and model scores to a 5-class integer system:
    -2: Very Negative, -1: Negative, 0: Neutral, 1: Positive, 2: Very Positive
    """
    # 1. Map Human Labels
    label_map = {
        'very negative': -2, 'negative': -1, 'neutral': 0, 
        'positive': 1, 'very positive': 2,
        'Very Negative': -2, 'Negative': -1, 'Neutral': 0, 
        'Positive': 1, 'Very Positive': 2
    }
    
    # Convert to string first to ensure consistent mapping, then map
    df['true_label'] = df['human_label_sentiment'].astype(str).str.strip().map(label_map)
    
    # Drop if mapping failed
    df = df.dropna(subset=['true_label'])

    # 2. Map Model Scores to 5 Classes (for Confusion Matrix/F1)
    def score_to_class(score):
        if pd.isna(score): return np.nan
        if score < -0.6: return -2
        elif score < -0.2: return -1
        elif score <= 0.2: return 0
        elif score <= 0.6: return 1
        else: return 2

    df['model_a_pred_class'] = df['one_step_llm_sentiment_score'].apply(score_to_class)
    df['model_b_pred_class'] = df['two_step_llm_sentiment_score'].apply(score_to_class)
    
    # Keep original scores for ROC AUC calculation
    df['model_a_score'] = df['one_step_llm_sentiment_score']
    df['model_b_score'] = df['two_step_llm_sentiment_score']

    # Ensure integer types for classes
    df['true_label'] = df['true_label'].astype(int)
    df['model_a_pred_class'] = df['model_a_pred_class'].astype(int)
    df['model_b_pred_class'] = df['model_b_pred_class'].astype(int)

    return df

def get_classification_metrics(y_true, y_pred):
    target_names = ['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos']
    labels = [-2, -1, 0, 1, 2]
    report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()

def calculate_roc_auc(y_true, y_score, labels=[-2, -1, 0, 1, 2]):
    """
    Calculates ROC AUC for multi-class using One-vs-Rest strategy.
    """
    # Convert to numpy arrays
    y_score_vals = y_score.values if hasattr(y_score, 'values') else np.array(y_score)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calculate ROC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score_vals)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    # To match dimensions: ravel y_true_bin (N*C) and tile y_score (N*C)
    y_true_bin_raveled = y_true_bin.ravel()
    y_score_tiled = np.tile(y_score_vals, n_classes)
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin_raveled, y_score_tiled)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return fpr, tpr, roc_auc

def analyze_by_group(df, group_column, model_col_pred, true_col='true_label'):
    results = []
    groups = df[group_column].unique()
    labels = [-2, -1, 0, 1, 2]
    
    for group in groups:
        subset = df[df[group_column] == group]
        if len(subset) < 5: continue 
        
        f1 = f1_score(subset[true_col], subset[model_col_pred], average='weighted', labels=labels, zero_division=0)
        results.append({
            'Group': group,
            'F1_Score': f1,
            'Count': len(subset)
        })
        
    return pd.DataFrame(results)

def add_bar_labels(ax, bars, fmt='{:.2f}'):
    """
    Adds text labels on top of bar charts.
    """
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, color='black')

def plot_combined_analysis(df):
    """
    Creates a single comprehensive figure with 6 visualizations:
    1. F1 Bar Chart (with labels)
    2. Confusion Matrix: One-Step
    3. Confusion Matrix: Two-Step
    4. ROC Curve: One-Step
    5. ROC Curve: Two-Step
    6. F1 by Source/Language (with labels)
    """
    
    # --- Calculate Metrics ---
    metrics_a = get_classification_metrics(df['true_label'], df['model_a_pred_class'])
    metrics_b = get_classification_metrics(df['true_label'], df['model_b_pred_class'])
    
    labels = [-2, -1, 0, 1, 2]
    target_names = ['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos']
    
    # --- Start Plotting ---
    fig = plt.figure(figsize=(20, 12)) 
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # 1. F1 Score Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    classes = ['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos', 'weighted avg']
    classes_present = [c for c in classes if c in metrics_a.index]
    
    f1_a = metrics_a.loc[classes_present, 'f1-score']
    f1_b = metrics_b.loc[classes_present, 'f1-score']
    
    x = np.arange(len(classes_present))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, f1_a, width, label='One-Step', color='#4C72B0')
    bars2 = ax1.bar(x + width/2, f1_b, width, label='Two-Step', color='#DD514C')
    
    # Add labels to F1 Comparison
    add_bar_labels(ax1, bars1)
    add_bar_labels(ax1, bars2)
    
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Comparison', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes_present, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.15) # Increased ylim slightly to fit labels

    # 2. Confusion Matrices (Top Middle & Right)
    ax2 = fig.add_subplot(gs[0, 1])
    cm_a = confusion_matrix(df['true_label'], df['model_a_pred_class'], labels=labels)
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                xticklabels=['V. Neg','Neg','Neu','Pos','V. Pos'],
                yticklabels=['V. Neg','Neg','Neu','Pos','V. Pos'])
    ax2.set_title('Confusion Matrix: One-Step', fontsize=12)
    ax2.set_ylabel('True (Human Labels)', labelpad=10)
    ax2.set_xlabel('Prediction', labelpad=10)

    ax3 = fig.add_subplot(gs[0, 2])
    cm_b = confusion_matrix(df['true_label'], df['model_b_pred_class'], labels=labels)
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=ax3,
                xticklabels=['V. Neg','Neg','Neu','Pos','V. Pos'],
                yticklabels=['V. Neg','Neg','Neu','Pos','V. Pos'])
    ax3.set_title('Confusion Matrix: Two-Step', fontsize=12)
    ax3.set_ylabel('True (Human Labels)', labelpad=10)
    ax3.set_xlabel('Prediction', labelpad=10)

    # 3. ROC Curves (Bottom Row)
    ax4 = fig.add_subplot(gs[1, 0])
    fpr_a, tpr_a, roc_auc_a = calculate_roc_auc(df['true_label'], df['model_a_score'], labels)
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(len(labels)), colors):
        ax4.plot(fpr_a[i], tpr_a[i], color=color, lw=2,
                 label=f'{target_names[i]} (AUC = {roc_auc_a[i]:0.2f})')
    
    ax4.plot([0, 1], [0, 1], 'k--', lw=2)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve: One-Step Model', fontsize=12)
    ax4.legend(loc="lower right", fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    fpr_b, tpr_b, roc_auc_b = calculate_roc_auc(df['true_label'], df['model_b_score'], labels)
    
    # Reset color cycle
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(len(labels)), colors):
        ax5.plot(fpr_b[i], tpr_b[i], color=color, lw=2,
                 label=f'{target_names[i]} (AUC = {roc_auc_b[i]:0.2f})')
    
    ax5.plot([0, 1], [0, 1], 'k--', lw=2)
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve: Two-Step Model', fontsize=12)
    ax5.legend(loc="lower right", fontsize=8)

    # 4. F1 by Source/Language (Bottom Right)
    ax6 = fig.add_subplot(gs[1, 2])
    group_col = 'source' if 'source' in df.columns else 'language'
    
    if group_col in df.columns:
        res_a = analyze_by_group(df, group_col, 'model_a_pred_class')
        res_b = analyze_by_group(df, group_col, 'model_b_pred_class')
        
        if not res_a.empty and not res_b.empty:
            merged = pd.merge(res_a, res_b, on='Group', suffixes=('_A', '_B'))
            merged['Group'] = merged['Group'].astype(str).str.capitalize()
            
            x_g = np.arange(len(merged['Group']))
            
            bars_g1 = ax6.bar(x_g - width/2, merged['F1_Score_A'], width, label='One-Step', color='#4C72B0')
            bars_g2 = ax6.bar(x_g + width/2, merged['F1_Score_B'], width, label='Two-Step', color='#DD514C')
            
            # Add labels to Group Comparison
            add_bar_labels(ax6, bars_g1)
            add_bar_labels(ax6, bars_g2)
            
            ax6.set_xticks(x_g)
            ax6.set_xticklabels(merged['Group'], rotation=45, ha='right')
            ax6.set_title(f'F1 by {group_col.capitalize()}', fontsize=12)
            ax6.legend(fontsize=8)
            ax6.set_ylim(0, 1.15)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data for group analysis', ha='center', va='center')
            ax6.axis('off')
    else:
        ax6.text(0.5, 0.5, 'No group column found', ha='center', va='center')
        ax6.axis('off')

    plt.savefig('newspaper_analysis/combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading data...")
    df = load_data()
    df = preprocess_labels(df)
    
    print(f"Data loaded: {len(df)} valid entries.")
    print(f"Label distribution:\n{df['true_label'].value_counts().sort_index()}")
    
    print("\nCalculating metrics and generating combined visualization...")
    plot_combined_analysis(df)
    
    # Print simple text summary for quick reference
    metrics_a = get_classification_metrics(df['true_label'], df['model_a_pred_class'])
    metrics_b = get_classification_metrics(df['true_label'], df['model_b_pred_class'])
    
    print("\n--- Weighted F1 Scores ---")
    print(f"One-Step Model: {metrics_a.loc['weighted avg', 'f1-score']:.3f}")
    print(f"Two-Step Model: {metrics_b.loc['weighted avg', 'f1-score']:.3f}")

if __name__ == "__main__":
    main()