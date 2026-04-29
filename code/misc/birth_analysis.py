# birth_analysis.py
#
# Analyzes instance creation dates to examine whether the Twitter/Elon Musk exodus
# (post-Oct 2022) led to a wave of new Mastodon instances. Compares the rule topic
# distributions of pre- and post-Oct 2022 instances using normalized category counts.
#
# Input:  data/instance_births_full.csv    (instance creation dates)
#         data/llm_category_analysis_data.csv  (GPT rule topic categories per instance)
# Output: bar plot of category distributions before/after Oct 2022 (plt.show only)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import scipy.stats as stats

# Load instance birth dates collected from /api/v1/instance contact_account.created_at
births = pd.read_csv(r'data\instance meta data\instance_births_full.csv')
print(len(births))
print(births.columns)
print(births.head())

# Convert 'birth' column to datetime, coerce errors (invalid dates will become NaT)
births['birth_parsed'] = pd.to_datetime(births['birth'], errors='coerce', utc=True)

cutoff_date = pd.Timestamp('2022-10-31T23:59:59', tz='UTC')

# Filter
after_oct_2022 = births[births['birth_parsed'] > cutoff_date]

# Results
print(f"Total instances: {len(births)}")
print(f"Instances born after Oct 2022: {len(after_oct_2022)}")
print(after_oct_2022.head())

# Merged with rules data 
rules = pd.read_csv(r'data\llm categorization\llm_category_analysis_data.csv')

merged = pd.merge(rules, births, on='Instance Name', how='inner')
print(len(merged))
print(merged.columns)

after_oct_2022_merged = merged[merged['birth_parsed'] > cutoff_date]
pre_oct_2022_merged = merged[merged['birth_parsed'] <= cutoff_date]

print(len(after_oct_2022_merged), len(pre_oct_2022_merged))

# For before and after, group by instance and get the count of each rule topic 
from collections import defaultdict, Counter

import re

def category_counts_per_instance(df, instance_col='Instance Name', cat_col='GPT category set'):
    """
    Returns:
      - dict: {instance -> {category -> count}}
      - df_out: DataFrame with one row per instance and a dict of counts
    """
    # 1) ensure string type
    tmp = df[[instance_col, cat_col]].copy()
    tmp[cat_col] = tmp[cat_col].astype(str)

    # helper: parse a single stringified set into list of category names
    def parse_set_string(s: str):
        # remove braces/quotes; split on commas; strip whitespace
        clean = re.sub(r"[{}\[\]'\"\u201C\u201D]", "", s)
        return [x.strip() for x in clean.split(",") if x.strip()]

    # 2–3) group by instance and collect strings
    grouped = (
        tmp.groupby(instance_col)[cat_col]
           .apply(list)                       # list of stringified sets per instance
           .reset_index(name='cat_strings')
    )

    # 4) convert collected strings -> flattened list -> Counter
    def to_counts(cat_strings):
        items = []
        for s in cat_strings:                  # each s is a string like "{'A', 'B'}" or "A, B"
            items.extend(parse_set_string(s))
        return dict(Counter(items))

    grouped['category_counts'] = grouped['cat_strings'].apply(to_counts)

    # optional: dictionary mapping
    result_dict = dict(zip(grouped[instance_col], grouped['category_counts']))

    # tidy output DataFrame (drop helper)
    df_out = grouped.drop(columns=['cat_strings'])

    return result_dict, df_out


after_dict, after_df = category_counts_per_instance(after_oct_2022_merged)
pre_dict, pre_df     = category_counts_per_instance(pre_oct_2022_merged)


#Plot the distribution of number of categories per instance for before and after Oct 2022
from collections import Counter

from collections import Counter

# --- Step 1: Aggregate category counts across all instances ---
after_counter = Counter()
for d in after_df['category_counts']:
    after_counter.update(d)

pre_counter = Counter()
for d in pre_df['category_counts']:
    pre_counter.update(d)

# --- Step 2: Convert to DataFrame ---
after_series = pd.Series(after_counter, name="After Oct 2022")
pre_series = pd.Series(pre_counter, name="Before Oct 2022")

category_counts = pd.concat([pre_series, after_series], axis=1).fillna(0).reset_index()
category_counts = category_counts.rename(columns={"index": "Category"})

# --- Step 3: Normalize by dataset size (number of instances) ---
after_size = len(after_df)
pre_size = len(pre_df)

category_counts["After Normalized"] = category_counts["After Oct 2022"] / after_size
category_counts["Before Normalized"] = category_counts["Before Oct 2022"] / pre_size

print(category_counts.head())

# --- Step 4: Plot normalized frequencies ---
plt.figure(figsize=(14, 8))
category_counts.set_index("Category")[["Before Normalized", "After Normalized"]].plot(
    kind="bar",
    figsize=(14, 8),
    alpha=0.8
)
plt.ylabel("Relative Frequency (per instance)")
plt.title("Normalized Distribution of Categories Before vs After Oct 2022")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

