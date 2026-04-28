# statistical_test_engvsnoneng.py
#
# Performs Welch's t-tests comparing English and non-English rules on three
# lexical features: word count, TTR, and rule count. Tests whether the two
# language groups differ significantly on these measures.
#
# Input:  df (DataFrame loaded externally with columns: 'language', 'word_count', 'ttr', 'rule_count')
# Output: printed t-statistics and p-values for each feature (no file output)

import pandas as pd
from scipy import stats

# Separate English and non-English subsets from the pre-loaded DataFrame
# Filtering English and Non-English rules
english_rules = df[df['language'] == 'English']
non_english_rules = df[df['language'] == 'Non-English']

# Function to perform t-test
def perform_t_test(feature):
    t_stat, p_value = stats.ttest_ind(english_rules[feature], non_english_rules[feature], equal_var=False)
    return t_stat, p_value

# Conduct t-tests
features = ['word_count', 'ttr', 'rule_count']
results = {feature: perform_t_test(feature) for feature in features}

# Print results
for feature, (t_stat, p_value) in results.items():
    print(f"T-test for {feature}:")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}\n")