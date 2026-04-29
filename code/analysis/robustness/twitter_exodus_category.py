# twitter_exodus_category.py
#
# Analyzes the stability of GPT-assigned rule topic categories across three
# Wayback Machine snapshot periods (pre-Elon, 2023, 2024). Cleans raw GPT
# output, converts categories to sets, and computes Jaccard similarity between
# periods. Reports proportion of rules with stable category assignments.
#
# Input:  data/chunks/rule_category_comparison.csv
#         (GPT categories per rule per time period from wayback_rule_categories.py)
# Output: printed similarity statistics and counts (no file output; save commented out)

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
# 11. GPT category cleaning
# ==============================
# Load rule category comparison; normalize whitespace and encoding in column names
df_categories = pd.read_csv(r'data\wayback machine\rule_category_comparison.csv')
df_categories.columns = df_categories.columns.str.strip()
df_categories.columns = df_categories.columns.str.replace(r'\s+', ' ', regex=True)
df_categories.columns = df_categories.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)

def parse_gpt_category(val):
    if pd.isna(val): return ""
    val = str(val)
    try:
        fixed = re.sub(r'([{,]\s*)([A-Za-z /]+):', r'\1"\2":', val)
        parsed = ast.literal_eval(fixed)
    except Exception:
        return val
    cats = [re.sub(r"^A:\s*","",str(k)).strip() for k in parsed.keys()]
    return ", ".join(cats)

gpt_cols = [c for c in df_categories.columns if "GPT category" in c]

for col in gpt_cols:
    df_categories[col] = df_categories[col].apply(parse_gpt_category)

print("Cleaned GPT category columns:")
print(df_categories[gpt_cols].head(10))
df_categories = df_categories.dropna(subset=gpt_cols, how='all')

#remove row if GPT category has Not Applicable 
df_categories = df_categories[~df_categories[gpt_cols].apply(lambda row: row.astype(str).str.contains("Not Applicable", case=False).any(), axis=1)]
print("After removing 'Not Applicable':", len(df_categories))

# ==============================
# 12. Category comparisons
# ==============================
# df_categories["changed_preelon_postelon2023"] = df_categories["GPT category preelon"] != df_categories["GPT category postelon2023"]
# df_categories["changed_postelon2023_postelon2024"] = df_categories["GPT category postelon2023"] != df_categories["GPT category postelon2024"]

# print("Number of category changes:")
# print("→ Preelon → Postelon2023:", df_categories["changed_preelon_postelon2023"].sum())
# print("→ Postelon2023 → Postelon2024:", df_categories["changed_postelon2023_postelon2024"].sum())

# print("\n=== Examples Preelon → Postelon2023 ===")
# print(df_categories.loc[df_categories["changed_preelon_postelon2023"],
#              ["instance","rule_preelon","GPT category preelon","rule_postelon2023","GPT category postelon2023"]].head(10))

# print("\n=== Examples Postelon2023 → Postelon2024 ===")
# print(df_categories.loc[df_categories["changed_postelon2023_postelon2024"],
#              ["instance","rule_postelon2023","GPT category postelon2023","rule_postelon2024","GPT category postelon2024"]].head(10))

# print("\nProportion of category changes:")
# print("Preelon → Postelon2023:", df_categories["changed_preelon_postelon2023"].mean())
# print("Postelon2023 → Postelon2024:", df_categories["changed_postelon2023_postelon2024"].mean())

# df_categories.to_csv("rule_category_comparison_final(2).csv", index=False)


def to_category_set(val):
    """Convert string categories into a Python set."""
    if pd.isna(val) or val == "":
        return set()
    return {item.strip() for item in str(val).split(",") if item.strip()}

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0  # both empty = identical
    return len(set1 & set2) / len(set1 | set2)

# turn into sets
gpt_cols = ["GPT category preelon", "GPT category postelon2023", "GPT category postelon2024"]
for col in gpt_cols:
    df_categories[col + "_set"] = df_categories[col].apply(to_category_set)

# compute similarities
df_categories["sim_preelon_postelon2023"] = df_categories.apply(
    lambda r: jaccard_similarity(r["GPT category preelon_set"], r["GPT category postelon2023_set"]), axis=1
)
df_categories["sim_postelon2023_postelon2024"] = df_categories.apply(
    lambda r: jaccard_similarity(r["GPT category postelon2023_set"], r["GPT category postelon2024_set"]), axis=1
)

# mark as same if similarity ≥ 0.9
df_categories["changed_preelon_postelon2023"] = df_categories["sim_preelon_postelon2023"] < 0.9
df_categories["changed_postelon2023_postelon2024"] = df_categories["sim_postelon2023_postelon2024"] < 0.9

# summary
print("Category similarity (Jaccard-based):")
print("→ Preelon → Postelon2023:", (~df_categories["changed_preelon_postelon2023"]).mean())
print("→ Postelon2023 → Postelon2024:", (~df_categories["changed_postelon2023_postelon2024"]).mean())

# examples where similarity is high but not perfect
print("\n=== Near Matches Preelon → Postelon2023 (0.8 ≤ sim < 1.0) ===")
print(df_categories.loc[
    (df_categories["sim_preelon_postelon2023"] >= 0.8) & (df_categories["sim_preelon_postelon2023"] < 1.0),
    ["instance","GPT category preelon_set","GPT category postelon2023_set","sim_preelon_postelon2023"]
].head(10))

print("\n=== Near Matches Postelon2023 → Postelon2024 (0.8 ≤ sim < 1.0) ===")
print(df_categories.loc[
    (df_categories["sim_postelon2023_postelon2024"] >= 0.8) & (df_categories["sim_postelon2023_postelon2024"] < 1.0),
    ["instance","GPT category postelon2023_set","GPT category postelon2024_set","sim_postelon2023_postelon2024"]
].head(10))

# save
#df_categories.to_csv("rule_category_comparison_with_similarity.csv", index=False)

# Count instances with Jaccard similarity < 0.9
less_than_09_preelon_2023 = (df_categories["sim_preelon_postelon2023"] < 0.9).sum()
less_than_09_2023_2024 = (df_categories["sim_postelon2023_postelon2024"] < 0.9).sum()

print("\n=== Instances with Jaccard < 0.9 ===")
print(f"Preelon → Postelon2023: {less_than_09_preelon_2023}")
print(f"Postelon2023 → Postelon2024: {less_than_09_2023_2024}")

# Total number of instances considered
total_instances = len(df_categories)

# Count instances with Jaccard similarity < 0.9
less_than_09_preelon_2023 = (df_categories["sim_preelon_postelon2023"] < 0.9).sum()
less_than_09_2023_2024 = (df_categories["sim_postelon2023_postelon2024"] < 0.9).sum()

# Calculate percentages
pct_less_than_09_preelon_2023 = less_than_09_preelon_2023 / total_instances * 100
pct_less_than_09_2023_2024 = less_than_09_2023_2024 / total_instances * 100

print("\n=== Instances with Jaccard < 0.9 ===")
print(f"Preelon → Postelon2023: {less_than_09_preelon_2023} / {total_instances} "
      f"({pct_less_than_09_preelon_2023:.2f}%)")
print(f"Postelon2023 → Postelon2024: {less_than_09_2023_2024} / {total_instances} "
      f"({pct_less_than_09_2023_2024:.2f}%)")

