# robustness_twitter_exodus.py
#
# Statistical analysis of rule and word count changes across three Wayback
# Machine snapshot periods (pre-Elon, Jan 2023, Jan 2024). Runs one-sample
# t-tests against zero and bootstrapped 95% CIs for each difference metric.
# Also examines user count growth (jumps) across periods.
#
# Input:  non_personal_preelon_notnull.csv   (pre-Elon rules, filtered)
#         non_personal_preelon_2022.csv      (pre-Elon user counts)
#         non_personal_postelon_2024(1).csv  (post-Elon 2024 rules)
#         non_personal_postelon_2023(1).csv  (post-Elon 2023 rules)
# Output: printed descriptive statistics and test results (no file output)

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import requests
import os
import random
import re

# ==============================
# 1. Load CSVs
# ==============================
# Pre-Elon snapshot: Sep-Oct 2022
df_1 = pd.read_csv(r'data\wayback machine\non_personal_preelon_notnull.csv')       # Sept-Oct 2022 (Pre-Elon)
temp_df = pd.read_csv(r'data\wayback machine\non_personal_preelon_2022.csv')

# Merge user_count into df_1
temp_df = temp_df[['instance', 'user_count']]
df_1 = pd.merge(df_1, temp_df, on='instance', how='left')

df_2 = pd.read_csv(r'data\wayback machine\non_personal_postelon_2024(1).csv')     # Post-Elon 2024
df_4 = pd.read_csv(r'data\wayback machine\non_personal_postelon_2023(1).csv')     # Post-Elon 2023

# ==============================
# 2. Merge datasets step by step
# ==============================
df_1 = df_1.rename(columns={'timestamp': 'timestamp_preelon', 'rules': 'rules_preelon', 'user_count': 'user_count_preelon'})
df_2 = df_2.drop(columns=['period', 'note'])
df_2 = df_2.rename(columns={'timestamp': 'timestamp_postelon2024', 'rules': 'rules_postelon2024', 'user_count': 'user_count_postelon2024'})
df = pd.merge(df_1, df_2, on='instance', how='inner')

df_4 = df_4.drop(columns=['period', 'note'])
df_4 = df_4.rename(columns={'timestamp': 'timestamp_postelon2023', 'rules': 'rules_postelon2023', 'user_count': 'user_count_postelon2023'})
df = pd.merge(df, df_4, on='instance', how='inner')

# ==============================
# 4. Data Cleaning
# ==============================
# Remove rows where any of the rule columns are NaN or empty
df = df.replace("", np.nan)
df = df.dropna(subset=['rules_preelon', 'rules_postelon2023', 'rules_postelon2024'])

print(f"After cleaning NaN/empty rules: {len(df)} rows")
print(df[['instance', 'rules_preelon', 'rules_postelon2023', 'rules_postelon2024']].head())

# ==============================
# 5. Basic Statistics
# ==============================
preelon_count = df['rules_preelon'].notna().sum()
total_instances = df['instance'].nunique()
print(f"Instances with preelon rules: {preelon_count}")
print(f"Total instances: {total_instances}")
print(f"Coverage: {preelon_count / total_instances:.2%}")

# ==============================
# 6. Compare rules across time periods
# ==============================
df['changed_preelon_postelon2023'] = df['rules_preelon'] != df['rules_postelon2023']
df['changed_postelon2023_postelon2024'] = df['rules_postelon2023'] != df['rules_postelon2024']

print("Changes between periods:")
print("→ Preelon → Postelon 2023:", df['changed_preelon_postelon2023'].sum())
print("→ Postelon 2023 → Postelon 2024:", df['changed_postelon2023_postelon2024'].sum())

# ==============================
# 8. Build a comparison table
# ==============================
comparison_cols = [
    'instance',
    'rules_preelon',
    'rules_postelon2023',
    'rules_postelon2024'
]
comparison_df = df[comparison_cols].copy()
comparison_df['changed_preelon_postelon2023'] = df['changed_preelon_postelon2023']
comparison_df['changed_postelon2023_postelon2024'] = df['changed_postelon2023_postelon2024']

changed_rules_df = comparison_df[
    comparison_df[['changed_preelon_postelon2023', 'changed_postelon2023_postelon2024']].any(axis=1)
]
print("Number of instances with changes:", len(changed_rules_df))


# ==============================
# 9. User count jumps
# ==============================
user_cols = ['instance','user_count_preelon','user_count_postelon2023','user_count_postelon2024']
user_df = df[user_cols].dropna()

user_df[['user_count_preelon','user_count_postelon2023','user_count_postelon2024']] = (
    user_df[['user_count_preelon','user_count_postelon2023','user_count_postelon2024']].astype(int)
)
user_df['jump_preelon_to_2023'] = user_df['user_count_postelon2023'] - user_df['user_count_preelon']
user_df['jump_2023_to_2024'] = user_df['user_count_postelon2024'] - user_df['user_count_postelon2023']
user_df['jump_preelon_to_2024'] = user_df['user_count_postelon2024'] - user_df['user_count_preelon']
print(user_df.head(20))

# ==============================
# 10. Rule + Word counts
# ==============================
def parse_rules(rule_str):
    if pd.isna(rule_str):
        return []
    try:
        rules = ast.literal_eval(rule_str)
        if isinstance(rules, list):
            return [rule.get("text", "").strip() for rule in rules if isinstance(rule, dict)]
        return []
    except (ValueError, SyntaxError):
        return []

def get_rule_count(rules): return len(rules) if isinstance(rules, list) else 0
def get_word_count(rules): return sum(len(r.split()) for r in rules) if isinstance(rules, list) else 0

for col in ["rules_preelon","rules_postelon2023","rules_postelon2024"]:
    df[f"{col}_parsed"] = df[col].apply(parse_rules)
    df[f"{col}_rule_count"] = df[f"{col}_parsed"].apply(get_rule_count)
    df[f"{col}_word_count"] = df[f"{col}_parsed"].apply(get_word_count)

df["rule_count_diff_preelon_2023"] = df["rules_postelon2023_rule_count"] - df["rules_preelon_rule_count"]
df["rule_count_diff_2023_2024"] = df["rules_postelon2024_rule_count"] - df["rules_postelon2023_rule_count"]
df["word_count_diff_preelon_2023"] = df["rules_postelon2023_word_count"] - df["rules_preelon_word_count"]
df["word_count_diff_2023_2024"] = df["rules_postelon2024_word_count"] - df["rules_postelon2023_word_count"]

print(df[[
    "instance",
    "rules_preelon_rule_count", "rules_postelon2023_rule_count", "rules_postelon2024_rule_count",
    "rule_count_diff_preelon_2023", "rule_count_diff_2023_2024",
    "rules_preelon_word_count", "rules_postelon2023_word_count", "rules_postelon2024_word_count",
    "word_count_diff_preelon_2023", "word_count_diff_2023_2024"
]].head(10))

from scipy import stats

# ==============================
# Rule & Word Count Statistics
# ==============================
features = {
    "rule_count_diff_preelon_2023": df["rule_count_diff_preelon_2023"],
    "rule_count_diff_2023_2024": df["rule_count_diff_2023_2024"],
    "word_count_diff_preelon_2023": df["word_count_diff_preelon_2023"],
    "word_count_diff_2023_2024": df["word_count_diff_2023_2024"],
    "jump_preelon_to_2023": user_df["jump_preelon_to_2023"],
    "jump_2023_to_2024": user_df["jump_2023_to_2024"],
    "jump_preelon_to_2024": user_df["jump_preelon_to_2024"]
}

print("\n=== Descriptive Stats and Significance Tests ===")
for name, series in features.items():
    series = series.dropna()
    mean_val = series.mean()
    median_val = series.median()
    std_val = series.std()

    # one-sample t-test vs 0
    t_stat, p_val = stats.ttest_1samp(series, 0)

    print(f"\n{name}:")
    print(f"  Mean   = {mean_val:.2f}")
    print(f"  Median = {median_val:.2f}")
    print(f"  Std    = {std_val:.2f}")
    print(f"  t-test against 0: t = {t_stat:.2f}, p = {p_val:.4f}")


from scipy import stats
import numpy as np

def bootstrap_ci(data, n_bootstrap=10000, ci=95, random_state=42):
    """
    Bootstrap confidence interval for the mean.
    """
    rng = np.random.default_rng(random_state)
    boot_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

# ==============================
# Rule & Word Count Statistics + Bootstraps
# ==============================
features = {
    "rule_count_diff_preelon_2023": df["rule_count_diff_preelon_2023"],
    "rule_count_diff_2023_2024": df["rule_count_diff_2023_2024"],
    "word_count_diff_preelon_2023": df["word_count_diff_preelon_2023"],
    "word_count_diff_2023_2024": df["word_count_diff_2023_2024"],
    "jump_preelon_to_2023": user_df["jump_preelon_to_2023"],
    "jump_2023_to_2024": user_df["jump_2023_to_2024"],
    "jump_preelon_to_2024": user_df["jump_preelon_to_2024"]
}

print("\n=== Descriptive Stats, Significance Tests, and Bootstrapped CIs ===")
for name, series in features.items():
    series = series.dropna().values  # use numpy array for bootstrapping
    mean_val = series.mean()
    median_val = np.median(series)
    std_val = series.std(ddof=1)

    # one-sample t-test vs 0
    t_stat, p_val = stats.ttest_1samp(series, 0)

    # bootstrap CI
    ci_low, ci_high = bootstrap_ci(series, n_bootstrap=10000, ci=95)

    print(f"\n{name}:")
    print(f"  Mean   = {mean_val:.2f}")
    print(f"  Median = {median_val:.2f}")
    print(f"  Std    = {std_val:.2f}")
    print(f"  95% CI (bootstrap) = [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  t-test against 0: t = {t_stat:.2f}, p = {p_val:.4f}")
