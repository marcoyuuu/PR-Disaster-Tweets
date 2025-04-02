# HumAID Dataset Analysis

## Overview
This document provides a comprehensive analysis of the HumAID datasets, which contain tweets related to humanitarian crises and disasters.

## Directory Structure
```
HumAID Datasets
└── all_combined
    ├── ._Licensing.txt
    ├── ._Readme.txt
    ├── Licensing.txt
    ├── Readme.txt
    ├── all_dev.tsv
    ├── all_test.tsv
    └── all_train.tsv
└── event_type
    ├── ._Licensing.txt
    ├── ._Readme.txt
    ├── Licensing.txt
    ├── Readme.txt
    ├── earthquake_dev.tsv
    ├── earthquake_test.tsv
    ├── earthquake_train.tsv
    ├── fire_dev.tsv
    ├── fire_test.tsv
    ├── fire_train.tsv
    ├── flood_dev.tsv
    ├── flood_test.tsv
    ├── flood_train.tsv
    ├── hurricane_dev.tsv
    ├── hurricane_test.tsv
    └── hurricane_train.tsv
└── events_set1
    ├── ._Licensing.txt
    ├── ._Readme.txt
    ├── Licensing.txt
    ├── Readme.txt
    ├── canada_wildfires_2016
    │   ├── canada_wildfires_2016_dev.tsv
    │   ├── canada_wildfires_2016_test.tsv
    │   └── canada_wildfires_2016_train.tsv
    ├── cyclone_idai_2019
    │   ├── cyclone_idai_2019_dev.tsv
    │   ├── cyclone_idai_2019_test.tsv
    │   └── cyclone_idai_2019_train.tsv
    ├── ecuador_earthquake_2016
    │   ├── ecuador_earthquake_2016_dev.tsv
    │   ├── ecuador_earthquake_2016_test.tsv
    │   └── ecuador_earthquake_2016_train.tsv
    ├── hurricane_harvey_2017
    │   ├── hurricane_harvey_2017_dev.tsv
    │   ├── hurricane_harvey_2017_test.tsv
    │   └── hurricane_harvey_2017_train.tsv
    ├── hurricane_irma_2017
    │   ├── hurricane_irma_2017_dev.tsv
    │   ├── hurricane_irma_2017_test.tsv
    │   └── hurricane_irma_2017_train.tsv
    ├── hurricane_maria_2017
    │   ├── hurricane_maria_2017_dev.tsv
    │   ├── hurricane_maria_2017_test.tsv
    │   └── hurricane_maria_2017_train.tsv
    ├── hurricane_matthew_2016
    │   ├── greece_wildfires_2018
    │   │   ├── greece_wildfires_2018_dev.tsv
    │   │   ├── greece_wildfires_2018_test.tsv
    │   │   └── greece_wildfires_2018_train.tsv
    │   ├── hurricane_matthew_2016_dev.tsv
    │   ├── hurricane_matthew_2016_test.tsv
    │   ├── hurricane_matthew_2016_train.tsv
    │   └── maryland_floods_2018
    │       ├── maryland_floods_2018_dev.tsv
    │       ├── maryland_floods_2018_test.tsv
    │       └── maryland_floods_2018_train.tsv
    ├── italy_earthquake_aug_2016
    │   ├── italy_earthquake_aug_2016_dev.tsv
    │   ├── italy_earthquake_aug_2016_test.tsv
    │   └── italy_earthquake_aug_2016_train.tsv
    ├── kaikoura_earthquake_2016
    │   ├── kaikoura_earthquake_2016_dev.tsv
    │   ├── kaikoura_earthquake_2016_test.tsv
    │   └── kaikoura_earthquake_2016_train.tsv
    ├── puebla_mexico_earthquake_2017
    │   ├── puebla_mexico_earthquake_2017_dev.tsv
    │   ├── puebla_mexico_earthquake_2017_test.tsv
    │   └── puebla_mexico_earthquake_2017_train.tsv
    └── srilanka_floods_2017
        ├── srilanka_floods_2017_dev.tsv
        ├── srilanka_floods_2017_test.tsv
        └── srilanka_floods_2017_train.tsv
└── events_set2
    ├── ._Licensing.txt
    ├── ._Readme.txt
    ├── Licensing.txt
    ├── Readme.txt
    ├── california_wildfires_2018
    │   ├── ._california_wildfires_2018_train.tsv
    │   ├── california_wildfires_2018_dev.tsv
    │   ├── california_wildfires_2018_test.tsv
    │   └── california_wildfires_2018_train.tsv
    ├── hurricane_dorian_2019
    │   ├── hurricane_dorian_2019_dev.tsv
    │   ├── hurricane_dorian_2019_test.tsv
    │   └── hurricane_dorian_2019_train.tsv
    ├── hurricane_florence_2018
    │   ├── hurricane_florence_2018_dev.tsv
    │   ├── hurricane_florence_2018_test.tsv
    │   └── hurricane_florence_2018_train.tsv
    ├── kerala_floods_2018
    │   ├── kerala_floods_2018_dev.tsv
    │   ├── kerala_floods_2018_test.tsv
    │   └── kerala_floods_2018_train.tsv
    ├── midwestern_us_floods_2019
    │   ├── midwestern_us_floods_2019_dev.tsv
    │   ├── midwestern_us_floods_2019_test.tsv
    │   └── midwestern_us_floods_2019_train.tsv
    └── pakistan_earthquake_2019
        ├── pakistan_earthquake_2019_dev.tsv
        ├── pakistan_earthquake_2019_test.tsv
        └── pakistan_earthquake_2019_train.tsv
```

## all_combined
**Location**: `c:\Users\Marco\Uni\ML\Proyecto\Tweets datasets\extracted_data_analysis\all_combined\all_combined`
**Total Files**: 5
**Total Tweets**: 5

### Labels
- injured_or_dead_people
- requests_or_urgent_needs
- sympathy_and_support

## event_type
**Location**: `c:\Users\Marco\Uni\ML\Proyecto\Tweets datasets\extracted_data_analysis\event_type\event_type`
**Total Files**: 14
**Total Tweets**: 5

### Labels
- caution_and_advice
- infrastructure_and_utility_damage
- other_relevant_information
- requests_or_urgent_needs
- sympathy_and_support

## events_set1_47K
**Location**: `c:\Users\Marco\Uni\ML\Proyecto\Tweets datasets\extracted_data_analysis\events_set1_47K\events_set1`
**Total Files**: 13
**Total Tweets**: 43409

### Labels
- caution_and_advice
- displaced_people_and_evacuations
- infrastructure_and_utility_damage
- injured_or_dead_people
- missing_or_found_people
- not_humanitarian
- other_relevant_information
- requests_or_urgent_needs
- rescue_volunteering_or_donation_effort
- sympathy_and_support

### Event Details
| Event | Files | Tweets | Splits |
|-------|--------|---------|--------|
| canada_wildfires_2016 | 3 | 2242 | dev, test, train |
| cyclone_idai_2019 | 3 | 3933 | dev, test, train |
| ecuador_earthquake_2016 | 3 | 1563 | dev, test, train |
| hurricane_harvey_2017 | 3 | 9112 | dev, test, train |
| hurricane_irma_2017 | 3 | 9399 | dev, test, train |
| hurricane_maria_2017 | 3 | 7278 | dev, test, train |
| hurricane_matthew_2016 | 9 | 3911 | dev, test, train, dev, test, train, dev, test, train |
| italy_earthquake_aug_2016 | 3 | 1201 | dev, test, train |
| kaikoura_earthquake_2016 | 3 | 2195 | dev, test, train |
| puebla_mexico_earthquake_2017 | 3 | 2015 | dev, test, train |
| srilanka_floods_2017 | 3 | 560 | dev, test, train |

## events_set2_29K
**Location**: `c:\Users\Marco\Uni\ML\Proyecto\Tweets datasets\extracted_data_analysis\events_set2_29K\events_set2`
**Total Files**: 8
**Total Tweets**: 33075

### Labels
- caution_and_advice
- displaced_people_and_evacuations
- infrastructure_and_utility_damage
- injured_or_dead_people
- missing_or_found_people
- not_humanitarian
- other_relevant_information
- requests_or_urgent_needs
- rescue_volunteering_or_donation_effort
- sympathy_and_support

### Event Details
| Event | Files | Tweets | Splits |
|-------|--------|---------|--------|
| california_wildfires_2018 | 3 | 7376 | dev, test, train |
| hurricane_dorian_2019 | 3 | 7613 | dev, test, train |
| hurricane_florence_2018 | 3 | 6264 | dev, test, train |
| kerala_floods_2018 | 3 | 7984 | dev, test, train |
| midwestern_us_floods_2019 | 3 | 1880 | dev, test, train |
| pakistan_earthquake_2019 | 3 | 1958 | dev, test, train |

## Recommendations

### For Event-Specific Analysis
- Use the Event Type dataset for direct access to specific disaster types
- Leverage the pre-split train/dev/test files for model development

### For General Analysis
- Use the All Combined dataset for a complete view of all humanitarian tweets
- Consider using Events Set 1 and 2 for larger-scale analysis

### For Disaster-Specific Analysis
- Focus on the event_type directory for disaster-specific data
- Use the individual event directories in Set 1 and Set 2 for detailed event analysis
