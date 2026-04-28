# error_log_analysis.py
#
# Counts and categorizes data collection errors in the raw instance rules file:
# empty rule lists, unknown errors, and HTTP status-code failures. Useful for
# assessing retrieval quality before cleaning.
#
# Input:  instance_rules_20_strat_0306.csv  (raw rules with error entries)
# Output: printed error counts and status code breakdown (no file output)

import pandas as pd
from collections import Counter

# Load raw rules file that may contain error strings from the data collection step
# final_rules = pd.read_csv(r'data\final_rules.csv')
final_rules = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\instance_rules_20_strat_0306.csv', encoding='utf-8')
print(final_rules.columns)
final_rules = final_rules.rename(columns={'rule': 'rules'})
# Counting occurrences
count_empty = (final_rules["rules"] == "[]").sum()
count_unknown_error = final_rules["rules"].str.startswith("Unknown error").sum()
count_failed_retrieve = final_rules["rules"].str.startswith("Failed to retrieve").sum()

# Display results
print(f"Empty lists ([]): {count_empty}")
print(f"Unknown error: {count_unknown_error}")
print(f"Failed to retrieve: {count_failed_retrieve}")

failed_rules = final_rules[final_rules["rules"].str.startswith("Failed to retrieve", na=False)]
print(failed_rules)

import re

# Extract numbers following "Failed to retrieve"
failed_rules["status_code"] = failed_rules["rules"].str.extract(r"Failed to retrieve instance information. Status code: (\d+)")
print(Counter(failed_rules['status_code']).keys())

