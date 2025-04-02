import os
import tarfile
import pandas as pd
from typing import Dict, List, Optional, Set
from collections import defaultdict

def setup_environment() -> None:
    """Ensure required libraries are installed."""
    try:
        import pandas
        print("✓ pandas is already installed")
    except ImportError:
        print("Installing pandas...")
        os.system("pip install pandas")
        print("✓ pandas installed successfully")

def define_paths() -> tuple[str, List[tuple[str, str]], str]:
    """Define paths and variables for processing."""
    # Get the current directory where the script is located
    download_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the archives to process with their expected inner directory names
    archives_to_process = [
        ("HumAID_data_all_combined.tar.gz", "all_combined"),
        ("HumAID_data_event_type.tar.gz", "event_type"),
        ("HumAID_data_events_set1_47K.tar.gz", "events_set1"),
        ("HumAID_data_events_set2_29K.tar.gz", "events_set2")
    ]
    
    # Define the base directory for extracted data
    extract_base_dir = os.path.join(download_dir, "extracted_data_analysis")
    
    return download_dir, archives_to_process, extract_base_dir

def extract_archives(download_dir: str, archives_to_process: List[tuple[str, str]], extract_base_dir: str) -> Dict[str, str]:
    """Extract all archives and return a mapping of subdir names to their extract paths."""
    extracted_paths = {}
    
    # Create the base extraction directory
    os.makedirs(extract_base_dir, exist_ok=True)
    
    for archive_filename, expected_inner_dir in archives_to_process:
        # Determine clean subdir name from the archive filename
        subdir_name = archive_filename.replace("HumAID_data_", "").replace(".tar.gz", "")
        
        # Construct paths
        extract_path = os.path.join(extract_base_dir, subdir_name)
        archive_path = os.path.join(download_dir, archive_filename)
        
        # Create extraction directory
        os.makedirs(extract_path, exist_ok=True)
        
        try:
            if not os.path.exists(archive_path):
                print(f"⚠️ Warning: Archive not found: {archive_filename}")
                continue
                
            print(f"\n📦 Extracting {archive_filename}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            print(f"✓ Successfully extracted to: {extract_path}")
            
            # Update the path to include the inner directory if it exists
            inner_dir_path = os.path.join(extract_path, expected_inner_dir)
            if os.path.exists(inner_dir_path) and os.path.isdir(inner_dir_path):
                extracted_paths[subdir_name] = inner_dir_path
            else:
                extracted_paths[subdir_name] = extract_path
            
        except FileNotFoundError:
            print(f"❌ Error: Archive file not found: {archive_filename}")
        except tarfile.ReadError:
            print(f"❌ Error: Failed to read archive: {archive_filename}")
        except Exception as e:
            print(f"❌ Unexpected error while processing {archive_filename}: {str(e)}")
    
    return extracted_paths

def create_directory_tree(path: str, prefix: str = "", is_last: bool = True, exclude_patterns: Set[str] = {'._.', '.DS_'}) -> List[str]:
    """Create a visual directory tree structure."""
    output = []
    
    # Get the base name of the path
    basename = os.path.basename(path)
    
    # Skip files/directories that match exclude patterns
    if any(basename.startswith(pattern) for pattern in exclude_patterns):
        return output
    
    # Add current item to output
    connector = "└── " if is_last else "├── "
    output.append(f"{prefix}{connector}{basename}")
    
    # If it's a directory, process its contents
    if os.path.isdir(path):
        # Get contents excluding hidden files
        contents = [item for item in sorted(os.listdir(path))
                   if not any(item.startswith(pattern) for pattern in exclude_patterns)]
        
        # Process each item
        for i, item in enumerate(contents):
            item_path = os.path.join(path, item)
            new_prefix = prefix + ("    " if is_last else "│   ")
            output.extend(create_directory_tree(item_path, new_prefix, i == len(contents) - 1))
    
    return output

def analyze_event_directory(event_dir: str) -> dict:
    """Analyze contents of an event-specific directory."""
    analysis = {
        'name': os.path.basename(event_dir),
        'total_files': 0,
        'tsv_files': [],
        'total_tweets': 0,
        'labels': set(),
        'splits': []
    }
    
    if not os.path.isdir(event_dir):
        return analysis
    
    for root, _, files in os.walk(event_dir):
        for file in files:
            if file.startswith('._') or file == '.DS_Store':
                continue
                
            analysis['total_files'] += 1
            if file.endswith('.tsv'):
                file_path = os.path.join(root, file)
                analysis['tsv_files'].append(file)
                
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    analysis['total_tweets'] += len(df)
                    if 'class_label' in df.columns:
                        analysis['labels'].update(df['class_label'].unique())
                    
                    # Detect split type from filename
                    for split in ['train', 'test', 'dev']:
                        if split in file.lower():
                            analysis['splits'].append(split)
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    
    return analysis

def analyze_directory(subdir_name: str, extract_path: str) -> Dict:
    """Analyze the contents of an extracted directory."""
    print(f"\n🔍 Analyzing directory: {subdir_name}")
    analysis_results = {
        'name': subdir_name,
        'path': extract_path,
        'total_files': 0,
        'structure': [],
        'events': [],
        'total_tweets': 0,
        'unique_labels': set()
    }
    
    try:
        if not os.path.isdir(extract_path):
            print(f"⚠️ Path is not a directory: {extract_path}")
            return analysis_results
            
        # Create directory tree visualization
        analysis_results['structure'] = create_directory_tree(extract_path)
        print("\n📁 Directory Structure:")
        for line in analysis_results['structure']:
            print(line)
            
        # List contents
        contents = [f for f in os.listdir(extract_path) if not f.startswith('._') and f != '.DS_Store']
        analysis_results['total_files'] = len(contents)
        print(f"\nFound {len(contents)} items in directory")
        
        # Sample data structure from TSV files
        tsv_files = [f for f in contents if f.endswith('.tsv')]
        if tsv_files:
            sample_file = os.path.join(extract_path, tsv_files[0])
            print(f"\n📊 Reading sample data from: {tsv_files[0]}")
            try:
                df = pd.read_csv(sample_file, sep='\t', nrows=5)
                analysis_results['total_tweets'] += len(df)
                
                if 'class_label' in df.columns:
                    labels = df['class_label'].unique()
                    analysis_results['unique_labels'].update(labels)
                    print("\nUnique labels found:")
                    print(list(labels))
                
            except Exception as e:
                print(f"❌ Error reading TSV file: {str(e)}")
        
        # Detailed event analysis for event sets
        if "events_set" in subdir_name:
            event_dirs = [d for d in contents if os.path.isdir(os.path.join(extract_path, d))]
            print(f"\n🌟 Analyzing {len(event_dirs)} event directories...")
            
            for event_dir in event_dirs:
                event_path = os.path.join(extract_path, event_dir)
                event_analysis = analyze_event_directory(event_path)
                analysis_results['events'].append(event_analysis)
                analysis_results['total_tweets'] += event_analysis['total_tweets']
                analysis_results['unique_labels'].update(event_analysis['labels'])
            
            # Print event summaries
            print("\n📊 Event Summaries:")
            for event in analysis_results['events']:
                print(f"\n- {event['name']}:")
                print(f"  • Files: {event['total_files']}")
                print(f"  • Tweets: {event['total_tweets']}")
                print(f"  • Splits: {', '.join(event['splits'])}")
                if event['labels']:
                    print(f"  • Labels: {', '.join(sorted(event['labels']))}")
            
        return analysis_results
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return analysis_results

def main():
    """Main execution function."""
    print("🚀 Starting HumAID dataset extraction and analysis...")
    
    # Phase 1: Setup
    setup_environment()
    download_dir, archives_to_process, extract_base_dir = define_paths()
    
    # Phase 2: Extraction
    print("\n📦 Phase 2: Extracting archives...")
    extracted_paths = extract_archives(download_dir, archives_to_process, extract_base_dir)
    
    # Phase 3: Analysis
    print("\n🔍 Phase 3: Analyzing extracted data...")
    analysis_results = {}
    for subdir_name, extract_path in extracted_paths.items():
        analysis_results[subdir_name] = analyze_directory(subdir_name, extract_path)
    
    # Generate markdown report
    generate_markdown_report(analysis_results, extract_base_dir)
    
    print("\n✨ Extraction and analysis complete!")

def generate_markdown_report(analysis_results: Dict, base_dir: str) -> None:
    """Generate a detailed markdown report of the analysis."""
    report_path = os.path.join(os.path.dirname(base_dir), "dataset_analysis.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# HumAID Dataset Analysis\n\n")
        
        # Overview
        f.write("## Overview\n")
        f.write("This document provides a comprehensive analysis of the HumAID datasets, which contain tweets related to humanitarian crises and disasters.\n\n")
        
        # Directory Structure
        f.write("## Directory Structure\n")
        f.write("```\n")
        f.write("HumAID Datasets\n")
        for name, analysis in analysis_results.items():
            for line in analysis['structure']:
                f.write(line + "\n")
        f.write("```\n\n")
        
        # Dataset Details
        for name, analysis in analysis_results.items():
            f.write(f"## {name}\n")
            
            # Basic information
            f.write(f"**Location**: `{analysis['path']}`\n")
            f.write(f"**Total Files**: {analysis['total_files']}\n")
            f.write(f"**Total Tweets**: {analysis['total_tweets']}\n")
            
            if analysis['unique_labels']:
                f.write("\n### Labels\n")
                for label in sorted(analysis['unique_labels']):
                    f.write(f"- {label}\n")
            
            # Event-specific information
            if analysis['events']:
                f.write("\n### Event Details\n")
                f.write("| Event | Files | Tweets | Splits |\n")
                f.write("|-------|--------|---------|--------|\n")
                for event in analysis['events']:
                    splits = ', '.join(event['splits']) if event['splits'] else 'N/A'
                    f.write(f"| {event['name']} | {event['total_files']} | {event['total_tweets']} | {splits} |\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### For Event-Specific Analysis\n")
        f.write("- Use the Event Type dataset for direct access to specific disaster types\n")
        f.write("- Leverage the pre-split train/dev/test files for model development\n\n")
        
        f.write("### For General Analysis\n")
        f.write("- Use the All Combined dataset for a complete view of all humanitarian tweets\n")
        f.write("- Consider using Events Set 1 and 2 for larger-scale analysis\n\n")
        
        f.write("### For Disaster-Specific Analysis\n")
        f.write("- Focus on the event_type directory for disaster-specific data\n")
        f.write("- Use the individual event directories in Set 1 and Set 2 for detailed event analysis\n")

if __name__ == "__main__":
    main()
