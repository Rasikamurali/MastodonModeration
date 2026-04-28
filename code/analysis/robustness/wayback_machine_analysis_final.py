# wayback_machine_analysis_final.py
#
# Merges pre- and post-Elon Wayback Machine rule snapshots and analyzes
# changes in rule count, word count, and user count across three periods:
# Sep-Oct 2022 (pre-Elon), Jan 2023, and Jan 2024. Also loads GPT category
# comparisons and checks stability of rule topic assignments using Jaccard
# similarity. Saves a rule change comparison table.
#
# Input:  non_personal_preelon_notnull.csv   (pre-Elon rules, filtered)
#         non_personal_preelon(2).csv        (pre-Elon user counts)
#         non_personal_postelon_2024(1).csv  (post-Elon 2024 rules)
#         non_personal_postelon_2023(1).csv  (post-Elon 2023 rules)
#         data/chunks/rule_category_comparison.csv  (GPT categories across periods)
# Output: instance_rule_changes_comparison.csv  (rule change flags per instance)

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import requests
import os
import random

# ==============================
# 1. Load CSVs
# ==============================
# Pre-Elon snapshot: Sep-Oct 2022
df_1 = pd.read_csv(r'non_personal_preelon_notnull.csv')       # Sept-Oct 2022 (Pre-Elon)

temp_df = pd.read_csv(r'non_personal_preelon(2).csv')   

print(len(df_1), len(temp_df))
print(df_1.columns)
print(temp_df.columns)
# Only keep instance + user_count from temp_df
temp_df = temp_df[['instance', 'user_count']]

# Merge user_count into df_1 based on instance
df_1 = pd.merge(df_1, temp_df, on='instance', how='left')

print(df_1.head())


df_2 = pd.read_csv(r'non_personal_postelon_2024(1).csv')     # Post-Elon 2024
df_4 = pd.read_csv(r'non_personal_postelon_2023(1).csv')     # Post-Elon 2023

# # print(len(df_1), len(df_2), len(df_3), len(df_4))
print(len(df_1), len(df_2), len(df_4))

# ==============================
# 2. Merge datasets step by step
# ==============================
# df_1 = df_1.drop(columns=['period', 'note'])
df_1 = df_1.rename(columns={'timestamp': 'timestamp_preelon', 'rules': 'rules_preelon', 'user_count': 'user_count_preelon'})

df_2 = df_2.drop(columns=['period', 'note'])
df_2 = df_2.rename(columns={'timestamp': 'timestamp_postelon2024', 'rules': 'rules_postelon2024', 'user_count': 'user_count_postelon2024'})
# Merge df_1 and df_2 first, keep suffixes to avoid column collisions
df = pd.merge(df_1, df_2, on='instance', how='inner', suffixes=('_preelon', '_postelon2024'))
print(len(df))
print(df.columns)

# Merge with df_3 (collection 2024)
# df = pd.merge(df, df_3, on='instance', how='inner', suffixes=('', '_collection2024'))
# print(len(df))
# print(df.columns)

df_4 = df_4.drop(columns=['period', 'note'])
df_4 = df_4.rename(columns={'timestamp': 'timestamp_postelon2023', 'rules': 'rules_postelon2023', 'user_count': 'user_count_postelon2023'})
# Merge with df_4 (postelon 2023)
df = pd.merge(df, df_4, on='instance', how='inner', suffixes=('', '_postelon2023'))
print(len(df))
print(df.columns)


# ==============================
# 4. Data Cleaning
# ==============================

#Clean columns: remove all note columns 
df = df.drop(columns=['period', 'note'])
print(df.columns)

# ==============================
# 5. Basic Statistics
# ==============================

# Count how many instances have preelon rules
preelon_count = df['rules_preelon'].notna().sum()
total_instances = df['instance'].nunique()
print(f"Instances with preelon rules: {preelon_count}")
print(f"Total instances: {total_instances}")
print(f"Coverage: {preelon_count / total_instances:.2%}")

# ==============================
# 6. Compare rules across time periods
# ==============================

# Compare preelon vs postelon2023
df['changed_preelon_postelon2023'] = df['rules_preelon'] != df['rules_postelon2023']

# Compare postelon2023 vs postelon2024
df['changed_postelon2023_postelon2024'] = df['rules_postelon2023'] != df['rules_postelon2024']

# Print a preview of differences
print(df[['instance', 'rules_preelon', 'rules_postelon2023', 'rules_postelon2024',
          'changed_preelon_postelon2023', 'changed_postelon2023_postelon2024']].head())

# Summary counts
print("Changes between periods:")
print("→ Preelon → Postelon 2023:", df['changed_preelon_postelon2023'].sum())
print("→ Postelon 2023 → Postelon 2024:", df['changed_postelon2023_postelon2024'].sum())



# ==============================
# 7. Visualization of rule changes
# ==============================
changes = {
    "Postelon 2023": df['changed_preelon_postelon2023'].sum(),
    "Postelon 2024": df['changed_postelon2023_postelon2024'].sum()
    # "Collection 2024": df['changed_collection2024'].sum()
}

plt.bar(changes.keys(), changes.values())
plt.ylabel("Number of Instances Changed")
plt.title("Rule Changes Compared to Preelon")
plt.show()

# ==============================
# 8. Build a comparison table (for inspection)
# ==============================
comparison_cols = [
    'instance',
    'rules_preelon',
    'rules_postelon2023',
    'rules_postelon2024'
]

# Select only comparison columns
comparison_df = df[comparison_cols].copy()

# Add change flags
comparison_df['changed_preelon_postelon2023'] = comparison_df['rules_preelon'] != comparison_df['rules_postelon2023']
comparison_df['changed_postelon2023_postelon2024'] = comparison_df['rules_postelon2023'] != comparison_df['rules_postelon2024']

# Keep only rows where there was at least one change
changed_rules_df = comparison_df[
    comparison_df[['changed_preelon_postelon2023', 'changed_postelon2023_postelon2024']].any(axis=1)
]

print("Number of instances with changes:", len(changed_rules_df))
print(changed_rules_df.head(20))  # Show first 20

#Optionally save to CSV
changed_rules_df.to_csv(r'instance_rule_changes_comparison.csv', index=False)


##### 
# In-depth analysis of the rules using Wayback Machine snapshots
#####

# user count jumps 

"""
I want to see for each instance what is the user count in preelon vs postelon 2023 vs postelon 2024 for where both user counts are present
"""
user_cols = [
    'instance',
    'user_count_preelon',
    'user_count_postelon2023',
    'user_count_postelon2024'
]

user_df = df[user_cols].copy()

# Keep only rows where all user counts are available
user_df = user_df.dropna(subset=['user_count_preelon', 'user_count_postelon2023', 'user_count_postelon2024'])

# (Optional) Convert counts to integers if needed
user_df[['user_count_preelon', 'user_count_postelon2023', 'user_count_postelon2024']] = (
    user_df[['user_count_preelon', 'user_count_postelon2023', 'user_count_postelon2024']].astype(int)
)

# Calculate jumps
user_df['jump_preelon_to_2023'] = user_df['user_count_postelon2023'] - user_df['user_count_preelon']
user_df['jump_2023_to_2024'] = user_df['user_count_postelon2024'] - user_df['user_count_postelon2023']
user_df['jump_preelon_to_2024'] = user_df['user_count_postelon2024'] - user_df['user_count_preelon']

# Preview results
print(user_df.head(20))


import ast

import ast
import pandas as pd

def parse_rules(rule_str):
    """
    Convert the string representation of rules into a list of rule texts.
    If parsing fails or the value is NaN, return [].
    """
    if pd.isna(rule_str):
        return []
    try:
        rules = ast.literal_eval(rule_str)
        if isinstance(rules, list):
            return [rule.get("text", "").strip() for rule in rules if isinstance(rule, dict)]
        return []
    except (ValueError, SyntaxError):
        return []



def get_rule_count(rules):
    if isinstance(rules, list):
        return len(rules)
    return 0

def get_word_count(rules):
    if isinstance(rules, list):
        return sum(len(rule.get("text", "").split()) for rule in rules if isinstance(rule, dict))
    return 0


for col in ["rules_preelon", "rules_postelon2023", "rules_postelon2024"]:
    df[f"{col}_parsed"] = df[col].dropna().apply(parse_rules)
    df[f"{col}_rule_count"] = df[f"{col}_parsed"].apply(get_rule_count)
    df[f"{col}_word_count"] = df[f"{col}_parsed"].apply(get_word_count)

print(df['rules_postelon2023_parsed'])
# word counts 
"""
Here, we analysis the difference in the word count of a set of rules of an instance across different time periods: preelon vs postelon 2023, postelon 2023 vs postelon 2024 
"""


# rule counts 
"""
Here, we analysis the difference in the rule count of a set of rules of an instance across different time periods: preelon vs postelon 2023, postelon 2023 vs postelon 2024 
"""

# Differences in rule counts
df["rule_count_diff_preelon_2023"] = df["rules_postelon2023_rule_count"] - df["rules_preelon_rule_count"]
df["rule_count_diff_2023_2024"] = df["rules_postelon2024_rule_count"] - df["rules_postelon2023_rule_count"]

# Differences in word counts
df["word_count_diff_preelon_2023"] = df["rules_postelon2023_word_count"] - df["rules_preelon_word_count"]
df["word_count_diff_2023_2024"] = df["rules_postelon2024_word_count"] - df["rules_postelon2023_word_count"]

print(df[[
    "instance",
    "rules_preelon_rule_count", "rules_postelon2023_rule_count", "rules_postelon2024_rule_count",
    "rule_count_diff_preelon_2023", "rule_count_diff_2023_2024",
    "rules_preelon_word_count", "rules_postelon2023_word_count", "rules_postelon2024_word_count",
    "word_count_diff_preelon_2023", "word_count_diff_2023_2024"
]].head(10))


df_categories = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\chunks\rule_category_comparison.csv')
print(df_categories.columns)

# Strip whitespace from column names
df_categories.columns = df_categories.columns.str.strip()

import re

print([repr(c) for c in df_categories.columns])

gpt_cols = [
    "GPT category preelon",
    "GPT category postelon2023",
    "GPT category postelon2024"
]


import re

import pandas as pd
import re
import ast

# --- Example: load your file ---
# df_categories = pd.read_csv("your_file.csv")

def parse_gpt_category(val):
    """
    Cleans GPT category outputs of form:
    {Not Applicable: 3, A: Harassment, Hate Speech: 2}
    and returns clean string like:
    'Not Applicable, Harassment, Hate Speech'
    """
    if pd.isna(val):
        return ""

    val = str(val)

    # Try to fix format so it looks like a dict
    try:
        # Add quotes around keys for safe eval
        fixed = re.sub(r'([{,]\s*)([A-Za-z /]+):', r'\1"\2":', val)
        parsed = ast.literal_eval(fixed)
    except Exception:
        return val

    # Collect only the keys (categories), clean up
    cats = []
    for k in parsed.keys():
        clean_k = re.sub(r"^A:\s*", "", str(k)).strip()
        cats.append(clean_k)

    return ", ".join(cats)


# --- Clean GPT category columns ---
df_categories.columns = df_categories.columns.str.strip()
df_categories.columns = df_categories.columns.str.replace(r'\s+', ' ', regex=True)
df_categories.columns = df_categories.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)

gpt_cols = [c for c in df_categories.columns if "GPT category" in c]

for col in gpt_cols:
    df_categories[col] = df_categories[col].apply(parse_gpt_category)

# --- Preview results ---
print("Cleaned GPT category columns:")
print(df_categories[gpt_cols].head(10))



import pandas as pd

df_categories["changed_preelon_postelon2023"] = (
    df_categories["GPT category preelon"] != df_categories["GPT category postelon2023"]
)

df_categories["changed_postelon2023_postelon2024"] = (
    df_categories["GPT category postelon2023"] != df_categories["GPT category postelon2024"]
)

# Summary counts
print("Number of category changes:")
print("→ Preelon → Postelon2023:", df_categories["changed_preelon_postelon2023"].sum())
print("→ Postelon2023 → Postelon2024:", df_categories["changed_postelon2023_postelon2024"].sum())

# Show examples where changes happened
print("\n=== Examples Preelon → Postelon2023 ===")
print(df_categories.loc[df_categories["changed_preelon_postelon2023"],
             ["instance", "rule_preelon", "GPT category preelon", 
              "rule_postelon2023", "GPT category postelon2023"]].head(10))

print("\n=== Examples Postelon2023 → Postelon2024 ===")
print(df_categories.loc[df_categories["changed_postelon2023_postelon2024"],
             ["instance", "rule_postelon2023", "GPT category postelon2023",
              "rule_postelon2024", "GPT category postelon2024"]].head(10))


print("\nProportion of category changes:")
print("Preelon → Postelon2023:", df_categories["changed_preelon_postelon2023"].mean())
print("Postelon2023 → Postelon2024:", df_categories["changed_postelon2023_postelon2024"].mean())

