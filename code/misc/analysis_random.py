# analysis_random.py
#
# Processes raw GPT output by cleaning category strings, applying a normalization
# mapping for category name variants, and binary-encoding all categories as one-hot
# columns. Used to produce the main LLM category encoding dataset.
#
# Input:  sampled_categorized_rules_my_mastodon_gpt4omini_0326_wUserCount.csv
#         (GPT-categorized rules with user count)
# Output: temp2.csv  (binary-encoded category columns added)

import pandas as pd

# Load GPT categorization output for the full rule dataset
df = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\community_rules_data.csv')
print(len(df))
print(df.columns)

bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]

#group by Instance Name and get the first User Count value for each instance
df = df.groupby('Instance Name').agg(
    User_Count=('User Count', 'first')
).reset_index()


# Assign each instance to a user count bin
df['User Count Bin'] = pd.cut(df['User_Count'], bins=bins, labels=bin_labels, right=False)

#Show count for each USer Count Bin
print(df['User Count Bin'].value_counts().sort_index())

data = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\one_shot_llm_category_encoding.csv')

# I want to sift through data and only keep rows where Insntance Name and translated text match 
merged_df = pd.merge(data, df[['Instance Name', 'User Count Bin']], on='Instance Name', how='inner')
print(len(merged_df))

merged_df.to_csv(r'C:\Users\rasik\Documents\Independent Study\data\deduplicated_oneshot.csv', index=False)