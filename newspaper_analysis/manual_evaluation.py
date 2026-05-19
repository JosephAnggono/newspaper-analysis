import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def load_data():
    file_path = 'newspaper_analysis/copy_of_entity_sentiment_paragraphs_one_step_vs_two_step.csv'
    df = pd.read_csv(file_path)
    
    # Keep only rows with human labels and model scores
    df = df.dropna(subset=['human_label_sentiment', 'one_step_llm_sentiment_score', 'two_step_llm_sentiment_score'])

    return df

def preprocess_labels(df):
    """
    Maps human labels and model scores to a 5-class integer system:
    -2: Very Negative -1: Negative 0: Neutral 1: Positive 2: Very Positive
    """
    # 1. Map Human Labels
    label_map = {
        'very negative': -2, 'negative': -1, 'neutral': 0, 'positive': 1, 'very positive': 2,
    }
    
    # Convert to string first to ensure consistent mapping, then map
    df['true_label'] = df['human_label_sentiment'].astype(str).str.strip().map(label_map)
    
    # 2. Map Model Scores to 5 Classes
    # We split the -1 to 1 range into 5 bins
    def score_to_class(score):
        if pd.isna(score):
            return np.nan
        if score < -0.6: return -2
        elif score < -0.2: return -1
        elif score <= 0.2: return 0
        elif score <= 0.6: return 1
        else: return 2

    df['model_a_pred'] = df['one_step_llm_sentiment_score'].apply(score_to_class)
    df['model_b_pred'] = df['two_step_llm_sentiment_score'].apply(score_to_class)
    
    # Ensure integer types
    df['true_label'] = df['true_label'].astype(int)
    df['model_a_pred'] = df['model_a_pred'].astype(int)
    df['model_b_pred'] = df['model_b_pred'].astype(int)

    return df

def get_classification_metrics(y_true, y_pred):
    """
    Calculates metrics for the 5-class system.
    """
    target_names = ['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos']
    labels = [-2, -1, 0, 1, 2]
    
    report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    return df_report

def plot_f1_comparison(df_metrics_model_a, df_metrics_model_b, title="F1 Score Comparison by Class"):
    """
    Plots a bar chart comparing F1 scores of two models for each of the 5 classes.
    """
    classes = ['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos', 'weighted avg']
    
    # Ensure classes exist in the dataframe
    classes_present = [c for c in classes if c in df_metrics_model_a.index]
    
    f1_a = df_metrics_model_a.loc[classes_present, 'f1-score']
    f1_b = df_metrics_model_b.loc[classes_present, 'f1-score']
    
    x = np.arange(len(classes_present))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, f1_a, width, label='One-Step Model', color='#4C72B0')
    bars2 = ax.bar(x + width/2, f1_b, width, label='Two-Step Model', color='#DD514C')
    
    ax.set_ylabel('F1 Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes_present, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig('f1_comparison_5class.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots a confusion matrix for the 5 classes.
    """
    labels = [-2, -1, 0, 1, 2]
    tick_labels = ['V.Neg', 'Neg', 'Neu', 'Pos', 'V.Pos']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels,
                yticklabels=tick_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label (Human)')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}_5class.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_by_group(df, group_column, model_col_pred, true_col='true_label'):
    """
    Analyzes Weighted F1 scores grouped by a specific column.
    """
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

def plot_group_comparison(df, group_column, title_prefix="Weighted F1 by"):
    """
    Compares Model A and Model B performance across groups.
    """
    res_a = analyze_by_group(df, group_column, 'model_a_pred')
    res_b = analyze_by_group(df, group_column, 'model_b_pred')
    
    if res_a.empty or res_b.empty:
        print(f"Not enough data to plot comparison by {group_column}")
        return

    merged = pd.merge(res_a, res_b, on='Group', suffixes=('_OneStep', '_TwoStep'))
    
    x = np.arange(len(merged['Group']))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, merged['F1_Score_OneStep'], width, label='One-Step Model', color='#4C72B0')
    bars2 = ax.bar(x + width/2, merged['F1_Score_TwoStep'], width, label='Two-Step Model', color='#DD514C')
    
    ax.set_ylabel('Weighted F1 Score')
    ax.set_title(f'{title_prefix} {group_column.capitalize()}')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Group'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'group_comparison_{group_column}_5class.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading data...")
    df = load_data()
    df = preprocess_labels(df)
    
    print(f"Data loaded: {len(df)} valid entries with human labels.")
    print(f"Label distribution:\n{df['true_label'].value_counts().sort_index()}")
    
    print("\nCalculating overall metrics...")
    metrics_a = get_classification_metrics(df['true_label'], df['model_a_pred'])
    metrics_b = get_classification_metrics(df['true_label'], df['model_b_pred'])
    
    print("\n--- One-Step Model Classification Report ---")
    print(metrics_a[['precision', 'recall', 'f1-score', 'support']].round(3))
    
    print("\n--- Two-Step Model Classification Report ---")
    print(metrics_b[['precision', 'recall', 'f1-score', 'support']].round(3))
    
    print("\nGenerating visualizations...")
    
    # 1. Overall F1 Comparison
    plot_f1_comparison(metrics_a, metrics_b, title="Overall F1 Score Comparison (5-Class)")
    
    # 2. Confusion Matrices
    plot_confusion_matrix(df['true_label'], df['model_a_pred'], "One-Step Model")
    plot_confusion_matrix(df['true_label'], df['model_b_pred'], "Two-Step Model")
    
    # 3. Analysis by Language
    if 'language' in df.columns:
        print("Analyzing performance by Language...")
        plot_group_comparison(df, 'language', title_prefix="Weighted F1 by")
    
    # 4. Analysis by Source
    if 'source' in df.columns:
        print("Analyzing performance by Source...")
        plot_group_comparison(df, 'source', title_prefix="Weighted F1 by")

if __name__ == "__main__":
    main()