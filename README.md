# PR-Disaster-Tweets: Analysis of Public Perception and Media Coverage During Natural Disasters in Puerto Rico

## Project Overview
This project focuses on analyzing public perception and media coverage during natural disasters in Puerto Rico, with a particular emphasis on Hurricane Maria (2017), the 2020 earthquakes, and 2025 tsunami advisory events. The analysis combines multiple datasets, including HumAID, ISCRAM18, and custom-scraped datasets, to provide insights into disaster response patterns, public sentiment, and humanitarian needs.

## Repository Structure
```
PR-Disaster-Tweets/
├── datasets/                  # All datasets used in the project
│   ├── HumAID_maria_tweets/   # HumAID dataset files for Hurricane Maria
│   ├── ISCRAM_maria_tweets/   # ISCRAM dataset files for Hurricane Maria
│   ├── PR_Earthquake_Tweets_Jan2020/ # Custom-scraped dataset for January 2020 earthquakes
│   └── PR_Advisory_Tweets_Feb_2025/  # Custom-scraped dataset for February 2025 tsunami advisory
├── .venv/                     # Virtual environment for dependencies
├── CITATION.md                # Citation information
├── LICENSE.md                 # License information
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
```

## Dataset Details

### `/datasets/HumAID_maria_tweets/`
Contains annotated disaster data (hurricanes and earthquakes) from the [CrisisNLP HumAID Dataset (ICWSM 2021)](https://crisisnlp.qcri.org/humaid_dataset). This dataset includes tweets categorized into humanitarian themes and serves as the foundation for supervised thematic classification. Subcategories include:
- Caution and advice
- Displaced people and evacuations
- Infrastructure and utility damage
- Injured or dead people
- Not humanitarian
- Other relevant information
- Requests or urgent needs
- Rescue volunteering or donation effort
- Sympathy and support

### `/datasets/ISCRAM_maria_tweets/`
Includes files from the [ISCRAM 2018 dataset](https://arxiv.org/pdf/1805.05144) on Hurricane Maria. It contains tweet IDs and image URLs. Text was retrieved via hydration to analyze public perception during the event. Visualizations include:
- Engagement metrics
- Likes distribution
- Tweet length histograms and boxplots
- Word clouds

### `/datasets/PR_Earthquake_Tweets_Jan2020/`
A custom collection of tweets related to the **January 2020 earthquakes in Puerto Rico**. This dataset enables comparisons between past disasters and current social media reactions. Data was scraped using [Octoparse](https://www.octoparse.com/) with filters for keywords, dates, and geolocation. This dataset will be processed for sentiment analysis and misinformation detection. Visualizations include:
- Interaction metrics
- Language distribution
- Likes distribution
- Tweet length histograms and boxplots
- Word clouds

### `/datasets/PR_Advisory_Tweets_Feb_2025/`
A custom collection of tweets related to the **February 2025 tsunami advisory**. This dataset enables comparisons between past disasters and current social media reactions. Data was scraped using [Octoparse](https://www.octoparse.com/) with filters for keywords, dates, and geolocation. This dataset will be processed for sentiment analysis and misinformation detection. Visualizations include:
- Engagement metrics
- Language distribution
- Likes distribution
- Tweet length histograms and boxplots
- Word clouds

## Running the Analysis

### Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PR-Disaster-Tweets.git
cd PR-Disaster-Tweets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Processing and Analysis

#### HumAID Dataset
```bash
python datasets/HumAID_maria_tweets/maria_analysis/analyze_humaid.py
```

#### ISCRAM Dataset
```bash
python datasets/ISCRAM_maria_tweets/analyze_ISCAM_tweets.py
```

#### Earthquake Tweets (January 2020)
```bash
python datasets/PR_Earthquake_Tweets_Jan2020/analyze_Jan2020_tweets.py
```

#### Advisory Tweets (February 2025)
```bash
python datasets/PR_Advisory_Tweets_Feb_2025/analyze_Feb2025_tweets.py
```

## Example Visualizations

### Hurricane Maria Analysis
- **Label Distribution**: ![Label Distribution](datasets/HumAID_maria_tweets/maria_analysis/maria_label_distribution.png)
- **Data Splits**: ![Data Splits](datasets/HumAID_maria_tweets/maria_analysis/maria_splits_distribution.png)

### Earthquake Tweets Analysis
- **Likes Distribution**: ![Likes Distribution](datasets/PR_Earthquake_Tweets_Jan2020/likes_distribution.png)
- **Word Cloud**: ![Word Cloud](datasets/PR_Earthquake_Tweets_Jan2020/tweet_word_cloud.png)

### Advisory Tweets Analysis
- **Engagement Metrics**: ![Engagement Metrics](datasets/PR_Advisory_Tweets_Feb_2025/advisory_engagement_metrics.png)
- **Language Distribution**: ![Language Distribution](datasets/PR_Advisory_Tweets_Feb_2025/advisory_language_distribution.png)

## Project Context

Puerto Rico is highly vulnerable to hurricanes and earthquakes. While these events differ in nature, both generate significant media impact and intense social media responses. Public perception, misinformation spread, and emotional language during these events can vary greatly. This project aims to analyze and compare how Puerto Ricans react to hurricanes (e.g., Hurricane Maria) and earthquakes (e.g., January 2020 earthquakes) using social media text analysis, particularly on Twitter. This comparison will help identify communication patterns and risk perception differences for various threats.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- HumAID dataset
- ISCRAM18 dataset
- Contributors and researchers involved in data collection and analysis

## Citation
Citations are included in CITATION.md file.