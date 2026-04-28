# robustness_instance_topic_rule_variation.py
#
# Computes within-instance and across-instance variance in rule topic coverage,
# grouped by GPT-assigned instance topic categories. Tests whether instances of
# the same topic type share similar rule topics (low variance = consistent norms).
#
# Input:  data/sampled_annotations_GPT_category(4)_full.csv  (GPT instance topics)
#         data/llm_category_analysis_data.csv                 (GPT rule topics per instance)
# Output: two plots showing within- and across-instance variance (plt.show only)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import regex as re

# Load GPT-assigned instance topic annotations
topical_df = pd.read_csv(r'data\sampled_annotations_GPT_category(4)_full.csv')    
print(topical_df.columns)
print(len(topical_df))
print(topical_df.head())

GPT_category1 = []

#Cleaning the GPT category column to remove unnecessary characters
for row in topical_df['GPT category']: 
    GPT_category1.append(re.sub(r'A:\s*|[\[\]\'"]', '', row).strip())
    
topical_df['GPT category cleaned'] = GPT_category1

print(topical_df.head())

topical_df = topical_df.drop(columns=['Unnamed: 0'])

rule_topic_data = pd.read_csv(r'data\llm_category_analysis_data.csv')

print(rule_topic_data.columns)
print(len(rule_topic_data))

combined_data = pd.merge(rule_topic_data, topical_df, on='Instance Name', how='inner')
print(len(combined_data))
print(combined_data.columns)    
print(combined_data.head())

######### 
"""
First, group the data by Instance Name. Get set of the rule topics. 
Then, group the data by topic categories assigned by GPT. 
Topic -> Instanceas -> Rule topic sets

"""

# Step 1: Split categories on commas
topical_df['GPT category split'] = topical_df['GPT category cleaned'].apply(lambda x: [cat.strip() for cat in x.split(',')])

# Step 2: Explode the DataFrame so each row is (Instance Name, single GPT category)
topical_exploded_df = topical_df.explode('GPT category split')

# Optional rename for clarity
topical_exploded_df = topical_exploded_df.rename(columns={'GPT category split': 'GPT category single'})

# Step 3: Merge again with rule_topic_data on Instance Name
combined_data_exploded = pd.merge(rule_topic_data, topical_exploded_df, on='Instance Name', how='inner')

from collections import defaultdict

# Step 4: Group: {GPT topic -> instances}
gpt_category_to_instances = defaultdict(list)

for idx, row in combined_data_exploded.iterrows():
    gpt_cat = row['GPT category single']
    instance = row['Instance Name']
    gpt_category_to_instances[gpt_cat].append(instance)

# Step 5: Group: {GPT topic -> rule topic set}
gpt_category_to_rule_topics = {}

for gpt_cat, instances in gpt_category_to_instances.items():
    rule_topics = set()
    
    for instance in instances:
        rt_rows = rule_topic_data[rule_topic_data['Instance Name'] == instance]
        for rt in rt_rows['GPT category set']:
            rule_topics.add(rt)
    
    gpt_category_to_rule_topics[gpt_cat] = rule_topics

summary_df = pd.DataFrame([
    {
        'GPT category': cat,
        'Num Instances': len(set(gpt_category_to_instances[cat])),
        'Rule topics': list(rule_topics)
    }
    for cat, rule_topics in gpt_category_to_rule_topics.items()
])

print(summary_df.head())


# Count of Rule Topics per GPT Topic
rule_topic_counts_per_gpt = (
    combined_data_exploded
    .groupby(['GPT category single', 'GPT category set'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['GPT category single', 'Count'], ascending=[True, False])
)

from collections import Counter
from collections import Counter
import re

# --- STEP 1: Clean + Split Rule topic into lists ---
def extract_rule_topics(value):
    if pd.isna(value):
        return []
    clean = re.sub(r'[\[\]\'"]', '', value)  # remove brackets and quotes
    return [item.strip() for item in clean.split(',') if item.strip()]

combined_data_exploded['Rule topic list'] = combined_data_exploded['GPT category set'].apply(extract_rule_topics)

# --- STEP 2: Explode the DataFrame on Rule topic list ---
final_df = combined_data_exploded.explode('Rule topic list').rename(columns={'Rule topic list': 'Rule topic single'})

# Count number of unique instances per GPT category
instances_per_gpt_category = final_df[['GPT category single', 'Instance Name']].drop_duplicates()
instance_counts = instances_per_gpt_category['GPT category single'].value_counts().to_dict()


# --- STEP 3: Group and Count ---
rule_topic_counts_per_gpt = (
    final_df
    .groupby(['GPT category single', 'Rule topic single'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['GPT category single', 'Count'], ascending=[True, False])
)

print(rule_topic_counts_per_gpt.head())

# Clean 'Rule topic single': remove curly braces, quotes, commas, and strip
def clean_topic(text):
    if pd.isna(text):
        return ''
    # Remove unwanted characters
    return re.sub(r"[{}\'\"“”]", '', text).replace(',', '').strip()

final_df['Rule topic single cleaned'] = final_df['Rule topic single'].apply(clean_topic)

# Re-count after cleaning
cleaned_rule_topic_counts_per_gpt = (
    final_df
    .groupby(['GPT category single', 'Rule topic single cleaned'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['GPT category single', 'Count'], ascending=[True, False])
)

print(cleaned_rule_topic_counts_per_gpt.head())

# Normalize by number of instances in each GPT category
cleaned_rule_topic_counts_per_gpt['Normalized Count'] = cleaned_rule_topic_counts_per_gpt.apply(
    lambda row: row['Count'] / instance_counts[row['GPT category single']],
    axis=1
)

min_val = cleaned_rule_topic_counts_per_gpt['Normalized Count'].min()
max_val = cleaned_rule_topic_counts_per_gpt['Normalized Count'].max()

cleaned_rule_topic_counts_per_gpt['Normalized 0-1'] = (
    (cleaned_rule_topic_counts_per_gpt['Normalized Count'] - min_val) / (max_val - min_val)
)


variance_df = (
    cleaned_rule_topic_counts_per_gpt
    .groupby("Rule topic single cleaned")["Normalized Count"]
    .var()
    .reset_index(name="Variance")
    .sort_values("Variance", ascending=False)
)

import numpy as np

# Pivot table: rows = instances, cols = norms, values = counts
pivot = final_df.pivot_table(
    index="Instance Name",
    columns="Rule topic single cleaned",
    values="GPT category single",
    aggfunc="count",
    fill_value=0
)

# 1. Within-instance variance (row-wise)
pivot["within_instance_variance"] = pivot.var(axis=1)

# 2. Across-instance variance (column-wise)
norm_variance = pivot.var(axis=0).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.histplot(pivot["within_instance_variance"], bins=30, kde=True)
plt.title("Distribution of Within-Instance Variance of Norms")
plt.xlabel("Variance across norms (per instance)")
plt.ylabel("Number of instances")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=norm_variance.index, y=norm_variance.values)
plt.xticks(rotation=45, ha="right")
plt.title("Across-Instance Variance of Norms")
plt.ylabel("Variance across instances")
plt.show()

