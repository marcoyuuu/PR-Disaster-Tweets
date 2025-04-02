import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns

def load_maria_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Hurricane Maria data from all splits."""
    # Path to Maria data
    maria_dir = os.path.join(base_dir, "extracted_data_analysis", "events_set1_47K", "events_set1", "hurricane_maria_2017")
    
    # Load each split
    train_df = pd.read_csv(os.path.join(maria_dir, "hurricane_maria_2017_train.tsv"), sep='\t')
    dev_df = pd.read_csv(os.path.join(maria_dir, "hurricane_maria_2017_dev.tsv"), sep='\t')
    test_df = pd.read_csv(os.path.join(maria_dir, "hurricane_maria_2017_test.tsv"), sep='\t')
    
    return train_df, dev_df, test_df

def analyze_maria_data(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Analyze the Maria dataset and return statistics."""
    stats = {
        'total_tweets': len(train_df) + len(dev_df) + len(test_df),
        'split_sizes': {
            'train': len(train_df),
            'dev': len(dev_df),
            'test': len(test_df)
        },
        'label_distribution': pd.concat([
            train_df['class_label'].value_counts(),
            dev_df['class_label'].value_counts(),
            test_df['class_label'].value_counts()
        ]).groupby(level=0).sum().to_dict()
    }
    
    return stats

def create_visualizations(stats: Dict, output_dir: str) -> None:
    """Create and save visualizations of the data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Split Distribution Pie Chart
    plt.figure(figsize=(10, 6))
    plt.pie(stats['split_sizes'].values(), labels=stats['split_sizes'].keys(), autopct='%1.1f%%')
    plt.title('Distribution of Tweets Across Splits')
    plt.savefig(os.path.join(output_dir, 'maria_splits_distribution.png'))
    plt.close()
    
    # 2. Label Distribution Bar Chart
    plt.figure(figsize=(15, 8))
    labels = list(stats['label_distribution'].keys())
    values = list(stats['label_distribution'].values())
    
    sns.barplot(x=values, y=labels)
    plt.title('Distribution of Tweet Labels')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'maria_label_distribution.png'))
    plt.close()

def generate_report(stats: Dict, output_dir: str) -> None:
    """Generate a markdown report of the analysis."""
    report_path = os.path.join(output_dir, "maria_analysis.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hurricane Maria Tweet Analysis\n\n")
        
        # Overview
        f.write("## Overview\n")
        f.write(f"Total tweets analyzed: {stats['total_tweets']}\n\n")
        
        # Split Distribution
        f.write("## Data Split Distribution\n")
        f.write("| Split | Number of Tweets | Percentage |\n")
        f.write("|-------|-----------------|------------|\n")
        for split, count in stats['split_sizes'].items():
            percentage = (count / stats['total_tweets']) * 100
            f.write(f"| {split} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        # Label Distribution
        f.write("## Label Distribution\n")
        f.write("| Label | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for label, count in sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_tweets']) * 100
            f.write(f"| {label} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        # Visualizations
        f.write("## Visualizations\n")
        f.write("### Split Distribution\n")
        f.write("![Split Distribution](maria_splits_distribution.png)\n\n")
        f.write("### Label Distribution\n")
        f.write("![Label Distribution](maria_label_distribution.png)\n\n")
        
        # Key Findings
        f.write("## Key Findings\n")
        
        # Find top 3 most common labels
        top_labels = sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
        f.write("### Most Common Tweet Categories:\n")
        for label, count in top_labels:
            percentage = (count / stats['total_tweets']) * 100
            f.write(f"- {label}: {count} tweets ({percentage:.1f}%)\n")
        
        # Calculate infrastructure and urgent needs related tweets
        infra_count = stats['label_distribution'].get('infrastructure_and_utility_damage', 0)
        urgent_count = stats['label_distribution'].get('requests_or_urgent_needs', 0)
        
        f.write("\n### Infrastructure and Urgent Needs:\n")
        f.write(f"- Infrastructure damage related tweets: {infra_count}\n")
        f.write(f"- Urgent needs related tweets: {urgent_count}\n")
        f.write(f"- Combined: {infra_count + urgent_count} tweets ")
        f.write(f"({((infra_count + urgent_count) / stats['total_tweets'] * 100):.1f}% of total)\n")

def main():
    """Main execution function."""
    print("🌀 Starting Hurricane Maria Tweet Analysis...")
    
    # Get the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "maria_analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        print("📥 Loading Maria dataset...")
        train_df, dev_df, test_df = load_maria_data(base_dir)
        
        # Analyze data
        print("🔍 Analyzing tweets...")
        stats = analyze_maria_data(train_df, dev_df, test_df)
        
        # Create visualizations
        print("📊 Creating visualizations...")
        create_visualizations(stats, output_dir)
        
        # Generate report
        print("📝 Generating analysis report...")
        generate_report(stats, output_dir)
        
        print(f"\n✨ Analysis complete! Results saved in: {output_dir}")
        print(f"Total tweets analyzed: {stats['total_tweets']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 