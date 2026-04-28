# fixing_regression_data.py
#
# Early, incomplete predecessor to regression_data_org.py. Builds a per-instance
# regression dataset with lexical features (word count, TTR), federation data,
# and instance birth dates. Most to_csv calls are commented out; this script is
# kept for reference. Use regression_data_org.py for the cleaned final version.
#
# Input:  data/community_rules_data.csv       (main rules dataset)
#         data/instance_births_full.csv        (instance birth dates)
#         federation_data_combined.csv         (federation peer counts)
#         data/llm_category_analysis_data.csv  (GPT category data)
#         temp2.csv  (additional category encodings)
# Output: regression_wTopicCounts.csv  (final merged regression dataset)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import os
import html
import regex as re

# Load the main rules dataset; rename column if needed
df = pd.read_csv(r'data/community_rules_data.csv')
df = df.rename({'instance': 'Instance Name'})
# df = df.rename(columns={'instance': 'Instance Name'})
print(df.columns)

# df= pd.read_csv(r'data\translated_rules_dataset.csv')
# print(len(df))
# df = df.dropna(subset='translated text')

rules= df['translated text']

# Filter rows where word count in 'translated text' is greater than 1
df = df[df['translated text'].str.split().str.len() > 1]

# # print(df)
# # # Filter rules with 7 or fewer words
# # filtered_rules = [rule for rule in rules if len(rule.split()) <=1]

# # print(len(filtered_rules))

# merged_data = pd.read_csv(r'final_instance_rule_count.csv')
# print(merged_data.columns)
# required_columns = ['Instance Name', 'rule count']
# merged_data = merged_data[required_columns]
# merged_df = pd.merge(df, merged_data, on='Instance Name', how='inner')
# print(merged_df.columns)
merged_df = df.copy()

columns = ['Instance Name', 'instance group', 'lang', 'translated text', 'User Count', 'rule count']

merged_df = merged_df[columns]
merged_df['User Count'] = pd.to_numeric(merged_df['User Count'], errors='coerce')

# print(merged_df['User Count'])

merged_df['Instance Name'] = merged_df['Instance Name'].str.strip().str.lower()
merged_df = merged_df[~merged_df['Instance Name'].isin(['libranet.de', 'venera.social'])]

# Define the bins and labels
bins = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]  
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$", r"$10^{7}$"]



# Assign each instance to a user count bin
merged_df['User Count Bin'] = pd.cut(merged_df['User Count'], bins=bins, labels=bin_labels, right=False)

# print("The data I am checking")
# print(merged_df.columns)
# print(merged_df.head())
# merge_more_data = merged_df.copy()
# print(merge_more_data.head())
# #Add federation- full_federation_data
# df1 = pd.read_csv('full_federation_data.csv')
# print(df1.columns)


# merge_more_data = pd.merge(merged_df, df1, on='Instance Name', how='inner')

# print(merge_more_data.columns)
# columns = ['Instance Name', 'Instance Id_x','instance group', 'lang', 'translated text', 'User Count_x', 'rule count', 'federating number', 'Description']

# merge_more_data = merge_more_data[columns]
# merge_more_data.rename(columns={'User Count_x': 'User Count'}, inplace=True)
# merge_more_data.rename(columns={'Instance Id_x': 'Instance Id'}, inplace=True)

merge_more_data = merged_df.copy()  

merge_more_data = merge_more_data.dropna(subset='translated text')

rules= merge_more_data['translated text']

# Filter rows where word count in 'translated text' is greater than 1
merge_more_data = merge_more_data[merge_more_data['translated text'].str.split().str.len() > 1]

# # print(df)
# Filter rules with 7 or fewer words
filtered_rules = [rule for rule in rules if len(rule.split()) <=1]

print(len(filtered_rules))

# Define the bins and labels
bins = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]  
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$", r"$10^{7}$"]

# Assign each instance to a user count bin
merge_more_data['User Count Bin'] = pd.cut(merge_more_data['User Count'], bins=bins, labels=bin_labels, right=False)

# Drop rows with missing translated text
merge_more_data = merge_more_data.dropna(subset='translated text')

print(merge_more_data.columns)

def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        # Lowercase the text
        text = text.lower()
        # Remove HTML entities
        text = html.unescape(text)
        # Remove punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        cleaned_list.append(text)
    return cleaned_list

cleaned_tt = clean_text(merge_more_data['translated text'])
merge_more_data['cleaned translated text'] = cleaned_tt

def word_count(data):
    wc = []
    for _, row in data.iterrows():
        text = row['cleaned translated text']
        words = re.findall(r'\b\w+\b', text)
        wc.append(len(words))  # Count only the valid words
    return wc

data_wc = word_count(merge_more_data)
merge_more_data['word count'] = data_wc 

print(merge_more_data.head())
print(merge_more_data.columns)

grouped_df = merge_more_data.groupby('Instance Name').agg({
    'word count': 'sum', 
    'User Count': 'first',
    'instance group': 'first',
    'lang': 'first',
    'cleaned translated text': lambda x: ' '.join(x),
    'rule count': 'first',
    'User Count Bin': 'first',
    'translated text': 'first', 
}).reset_index()


print(grouped_df.head())


grouped_df['filtered_text'] = grouped_df['cleaned translated text'].apply(
    lambda text: ' '.join([word for word in text.split() if len(word.split()) >=7])
)

# Define a function to calculate TTR
def calculate_ttr(text):
    words = text.split()  # Split text into words
    num_unique_words = len(set(words))  # Count unique words (types)
    total_words = len(words)  # Count all words (tokens)
    return num_unique_words / total_words if total_words > 0 else 0

# Create a new DataFrame with TTR for each group

grouped_df['TTR'] = grouped_df['cleaned translated text'].apply(calculate_ttr)


print(grouped_df.head())
print(grouped_df.columns)

# grouped_df.to_csv(r'regression_data_strat20.csv', index=False)

final_columns = [ 'Instance Name', 'word count', 'User Count', 'instance group', 
                 'rule count', 'User Count Bin','TTR']

grouped_df = grouped_df[final_columns]
#grouped_df.to_csv(r'regression_data_final_full.csv', index=False)


# df = pd.read_csv(r'topics_per_instance.csv')
# columns = ['Instance Name', 'Topics', 'User_Count', 'User Count Bin']
# df = df[columns]
# df.to_csv('topics_per_instance_final.csv', index=False)

# df = pd.read_csv(r'regression_data_final.csv')
# print(df.columns)


# w= pd.read_csv(r'instance_weekly_data.csv')
# print(w.head())

# b = pd.read_csv(r'instance_births.csv')
# print(b.head())

# df1 = pd.merge(df, w, on='Instance Name', how='inner')
# print(df1.head())

# df2 = pd.merge(df1, b, on='Instance Name', how='inner')
# print(df2.columns)
# print(len(df2))
# print(df2.head())

# df2 = df2.drop(columns=['Error'])
# print(df2.columns)
# #df2.to_csv(r'regression_data_final2.csv', index=False)

# print(len(df2))
new = pd.read_csv(r'data/instance_births_full.csv')
print(new.columns)
print(len(new))

new_df = pd.merge(grouped_df, new, on='Instance Name', how='inner')
print(new_df.columns)
print(len(new_df))
instance_names = new_df['Instance Name']
#new_df.to_csv(r'regression_data_final2_strat20.csv', index=False)

fed_new = pd.read_csv(r'federation_data_combined.csv')
print(fed_new.columns)
print(len(fed_new))
fed_new = fed_new[fed_new['Instance Name'].isin(instance_names)]
fed_new = fed_new.drop_duplicates(subset='Instance Name')
print(len(fed_new))
new_new = pd.merge(new_df, fed_new, on='Instance Name', how='inner')
print(new_new.columns)

print(len(new_new))
#new_new.to_csv(r'regression_data_final2_combinedfull_regbirth.csv', index=False)

print("Checking for topic counts")

updated_df1 = pd.read_csv(r'data\llm_category_analysis_data.csv')
print(updated_df1.columns)
print(len(updated_df1))
updated_df2 = pd.read_csv(r'temp2.csv')
print(updated_df2.columns)
print(len(updated_df2))
updated_df2 = updated_df2.rename(columns={'instance group_y': 'instance group', 'rule_y': 'rule', 'translated text_y': 'translated text'})
updated_df = pd.concat([updated_df1, updated_df2], axis=0)

print(updated_df.columns)
print(len(updated_df))
print(updated_df[['Instance Name', 'GPT category set', 'User Count', 'instance group', 'Category Count']])

import ast

# Ensure 'GPT category set' is properly formatted as sets
updated_df['GPT category set'] = updated_df['GPT category set'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


updated_df = updated_df.groupby('Instance Name').agg(
    Topics=('GPT category set', lambda x: {cat: sum(cat in s for s in x) for s in x for cat in s}),
    User_Count=('User Count', 'first'),
    instance_group=('instance group', 'first'),
    Topic_Count=('Category Count', 'sum')
).reset_index()


print(updated_df.columns)
print(updated_df.head())
print(len(updated_df))

print(len(new_new))

print(new_new.columns)  

another_new_new = pd.merge(new_new, updated_df, on='Instance Name', how='left')
print(another_new_new.columns)
print(len(another_new_new))

another_new_new = pd.merge(new_new, updated_df, on='Instance Name', how='left')
another_new_new = another_new_new.drop(columns = ['User_Count', 'User Count_y', 'instance_group_x', 'instance_group_y'])


print(another_new_new.columns)  # Check final columns
print(len(another_new_new))  # Should match len(new_new)


another_new_new.to_csv(r'regression_wTopicCounts.csv', index=False)
