import os
import pandas as pd
from typing import Dict, List

def load_and_combine_maria_data() -> pd.DataFrame:
    """Load and combine all Hurricane Maria tweets from all splits."""
    # Path to Maria data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    maria_dir = os.path.join(base_dir, "extracted_data_analysis", "events_set1_47K", 
                            "events_set1", "hurricane_maria_2017")
    
    # Load all splits
    dfs = []
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(maria_dir, f"hurricane_maria_2017_{split}.tsv")
        df = pd.read_csv(file_path, sep='\t')
        df['split'] = split  # Add split information
        dfs.append(df)
    
    # Combine all splits
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def save_tweets_by_category(df: pd.DataFrame, output_dir: str) -> None:
    """Save tweets grouped by their category."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all tweets in one file
    all_tweets_file = os.path.join(output_dir, "all_maria_tweets.csv")
    df.to_csv(all_tweets_file, index=False)
    print(f"✓ Saved all tweets to: {all_tweets_file}")
    
    # Save tweets by category
    for category in df['class_label'].unique():
        category_df = df[df['class_label'] == category]
        category_file = os.path.join(output_dir, f"maria_tweets_{category}.csv")
        category_df.to_csv(category_file, index=False)
        print(f"✓ Saved {len(category_df)} {category} tweets to: {category_file}")

def generate_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Generate a summary of the extracted tweets."""
    summary_file = os.path.join(output_dir, "maria_tweets_summary.md")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Hurricane Maria Tweets Summary\n\n")
        
        # Overall statistics
        f.write("## Dataset Overview\n")
        f.write(f"- Total tweets: {len(df):,}\n")
        f.write(f"- Unique categories: {len(df['class_label'].unique())}\n\n")
        
        # Distribution by category
        f.write("## Tweet Distribution by Category\n")
        f.write("| Category | Count | Example Tweet |\n")
        f.write("|----------|-------|---------------|\n")
        
        for category in sorted(df['class_label'].unique()):
            category_df = df[df['class_label'] == category]
            example_tweet = category_df['tweet_text'].iloc[0]
            if len(example_tweet) > 100:
                example_tweet = example_tweet[:97] + "..."
            f.write(f"| {category} | {len(category_df):,} | {example_tweet} |\n")
        
        # Distribution by split
        f.write("\n## Distribution by Split\n")
        split_dist = df['split'].value_counts()
        for split, count in split_dist.items():
            f.write(f"- {split}: {count:,} tweets ({count/len(df)*100:.1f}%)\n")

def main():
    """Main execution function."""
    print("🌀 Starting Hurricane Maria tweets extraction...")
    
    try:
        # Load and combine data
        print("📥 Loading Maria dataset...")
        df = load_and_combine_maria_data()
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "maria_tweets")
        
        # Save tweets
        print("\n💾 Saving tweets by category...")
        save_tweets_by_category(df, output_dir)
        
        # Generate summary
        print("\n📊 Generating summary...")
        generate_summary(df, output_dir)
        
        print(f"\n✨ Extraction complete! Files saved in: {output_dir}")
        print(f"Total tweets extracted: {len(df):,}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")

if __name__ == "__main__":
    main() 