import os
import pandas as pd

def examine_maria_data():
    """Examine the contents of Maria dataset files."""
    # Path to Maria data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    maria_dir = os.path.join(base_dir, "extracted_data_analysis", "events_set1_47K", 
                            "events_set1", "hurricane_maria_2017")
    
    # Load one file to check its structure
    train_file = os.path.join(maria_dir, "hurricane_maria_2017_train.tsv")
    
    print("📊 Examining Hurricane Maria dataset structure...")
    
    try:
        # Read the first few rows
        df = pd.read_csv(train_file, sep='\t', nrows=5)
        
        print("\n📋 Dataset columns:")
        print(df.columns.tolist())
        
        print("\n📝 First few entries:")
        print(df.to_string())
        
        # Count total tweets
        total_tweets = len(pd.read_csv(train_file, sep='\t'))
        print(f"\n📈 Total tweets in training set: {total_tweets}")
        
        print("\n🔍 Data Structure Analysis:")
        print("1. The dataset contains tweet IDs and labels only")
        print("2. To get the actual tweet text, you would need to:")
        print("   a) Use Twitter's API to hydrate the tweets")
        print("   b) Contact the dataset authors for the full dataset")
        print("   c) Check if there's an already hydrated version available")
        
        print("\n⚠️ Note: Many tweets may no longer be available on Twitter due to:")
        print("- Tweets being deleted")
        print("- Accounts being suspended or deleted")
        print("- Twitter's data retention policies")
        
    except Exception as e:
        print(f"❌ Error examining data: {str(e)}")

if __name__ == "__main__":
    examine_maria_data() 