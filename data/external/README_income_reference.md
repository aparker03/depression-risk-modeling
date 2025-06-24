# Census Connection

This folder contains the dataset `Income_race_edu_gender.csv`, which was used to explore connections between income, race, education, and gender in relation to depression risk modeling.

## Source
The data was compiled from U.S. Census Bureau datasets and cleaned locally to include relevant features only. It may be aggregated by demographic and socioeconomic categories.
Link: https://www.census.gov/library/publications/2024/demo/p60-282.html
Table name used: Table A-1. Income Summary Measures by Selected Characteristics: 2022 and 2023 [< 1.0 MB]

## File Details
- `Income_race_edu_gender.csv`: Contains grouped data on income levels across race, education level, and gender. Useful for exploring socioeconomic disparities related to mental health indicators.

## Usage
This data can be used to create new features or stratify models by demographic and economic characteristics. Consider using it alongside NHANES or PHQ-9 datasets for enriched insights.
How it was used in SL by Brandon Fox: Take Race, Gender, Education for each person and then connect it to the median income for that indicator and then take an average of their results. This creates an estimate for the participants to use during supervised learning.

