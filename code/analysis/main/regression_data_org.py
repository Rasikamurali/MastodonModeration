# regression_data_org.py
#
# Builds the per-instance regression dataset by computing lexical features
# (word count, TTR, FK readability) from translated rules, then merging with
# instance birth dates and federation counts. Output is the primary input
# for all regression analyses.
#
# Input:  data/community_rules_data.csv         (translated rules with user counts)
#         data/instance_births_full.csv          (instance creation dates)
#         data/federation_data_combined.csv      (federation peer counts)
# Output: data/regression_data_lexicalfeature_fed_birth.csv

import os
import html
import regex as re
import pandas as pd
import textstat

# Load and filter rules data
df = pd.read_csv(r'data/primary/community_rules_data.csv')
df = df.dropna(subset=['translated text'])
df = df[df['translated text'].str.split().str.len() > 1]

df['User Count'] = pd.to_numeric(df['User Count'], errors='coerce')
df['Instance Name'] = df['Instance Name'].str.strip().str.lower()

# Exclude instances with atypically long or legally sensitive rule sets
df = df[~df['Instance Name'].isin(['libranet.de', 'venera.social'])]

# Assign each rule to a log-scale user count bin
bins = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$", r"$10^{7}$"]
df['User Count Bin'] = pd.cut(df['User Count'], bins=bins, labels=bin_labels, right=False)

df = df[['Instance Name', 'instance group', 'lang', 'translated text', 'User Count', 'rule count', 'User Count Bin']]


def clean_text(text_list):
    cleaned = []
    for text in text_list:
        text = text.lower()
        text = html.unescape(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        cleaned.append(text)
    return cleaned


def word_count(data):
    return [len(re.findall(r'\b\w+\b', row['cleaned translated text'])) for _, row in data.iterrows()]


def calculate_ttr(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0


def fk_score(text):
    try:
        return textstat.flesch_reading_ease(str(text))
    except Exception:
        return None


df['cleaned translated text'] = clean_text(df['translated text'])
df['word count'] = word_count(df)

# Aggregate to one row per instance
instance_df = df.groupby('Instance Name').agg(
    word_count=('word count', 'sum'),
    User_Count=('User Count', 'first'),
    instance_group=('instance group', 'first'),
    lang=('lang', 'first'),
    rule_count=('rule count', 'first'),
    User_Count_Bin=('User Count Bin', 'first'),
    cleaned_text=('cleaned translated text', ' '.join),
    rule_set=('translated text', lambda x: ' '.join(
        [r.strip().rstrip('.') + '.' for r in x.astype(str)]
    ))
).reset_index()

instance_df['TTR'] = instance_df['cleaned_text'].apply(calculate_ttr)
instance_df['Fk Score'] = instance_df['rule_set'].apply(fk_score)

final_columns = ['Instance Name', 'word_count', 'User_Count', 'instance_group',
                 'rule_count', 'User_Count_Bin', 'TTR', 'Fk Score']
instance_df = instance_df[final_columns]

# Merge with instance birth dates
births_df = pd.read_csv(r'data/instance meta data/instance_births_full.csv')
instance_df = pd.merge(instance_df, births_df, on='Instance Name', how='inner')

# Merge with federation data, keeping one row per instance
fed_df = pd.read_csv(r'data/instance meta data/federation_combined.csv')
fed_df = fed_df[fed_df['Instance Name'].isin(instance_df['Instance Name'])]
fed_df = fed_df.drop_duplicates(subset='Instance Name')
instance_df = pd.merge(instance_df, fed_df, on='Instance Name', how='inner')

instance_df.to_csv(r'data/regression/regression_data_lexicalfeature_fed_birth.csv', index=False)
