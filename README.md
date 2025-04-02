# PR-Disaster-Tweets: Analysis of Public Perception and Media Coverage During Natural Disasters in Puerto Rico

## Project Overview
This project focuses on analyzing public perception and media coverage during natural disasters in Puerto Rico, with a particular emphasis on Hurricane Maria (2017) and the 2020 earthquakes. The analysis combines multiple datasets including HumAID and ISCRAM18 to provide insights into disaster response patterns, public sentiment, and humanitarian needs.

## Citation
If you use this project or the HumAID dataset in your research, please cite:

```bibtex
@inproceedings{humaid2020,
    Author = {Firoj Alam, Umair Qazi, Muhammad Imran, Ferda Ofli},
    Booktitle = {15th International Conference on Web and Social Media (ICWSM)},
    Keywords = {Social Media, Crisis Computing, Tweet Text Classification, Disaster Response},
    Title = {HumAID: Human-Annotated Disaster Incidents Data from Twitter},
    Year = {2021}
}
```

## Repository Structure
```
PR-Disaster-Tweets/
├── data/                      # Raw and processed datasets
│   ├── maria_tweets/         # Hurricane Maria tweets dataset
│   └── humaid/               # HumAID dataset files
├── src/                      # Source code
│   ├── data_processing/      # Scripts for data extraction and processing
│   └── analysis/            # Analysis and visualization scripts
├── notebooks/                # Jupyter notebooks for analysis
├── docs/                     # Documentation
└── results/                  # Analysis results and visualizations
```

## Dataset Details

### Hurricane Maria Tweets
- Total tweets: 7,278
- Categories:
  - Caution and advice
  - Displaced people and evacuations
  - Infrastructure and utility damage
  - Injured or dead people
  - Not humanitarian
  - Other relevant information
  - Requests or urgent needs
  - Rescue volunteering or donation effort
  - Sympathy and support

### Data Splits
- Training: 5,094 tweets (70.0%)
- Test: 1,442 tweets (19.8%)
- Development: 742 tweets (10.2%)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PR-Disaster-Tweets.git
cd PR-Disaster-Tweets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing
```bash
python src/data_processing/extract_maria_tweets.py
python src/data_processing/process_humaid.py
```

### Analysis
```bash
python src/analysis/check_maria_data.py
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- HumAID dataset
- ISCRAM18 dataset
- Contributors and researchers involved in data collection and analysis 