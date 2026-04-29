# counting_instance.py
#
# Counts the number of instances per log-scale user count bin and merges
# the community rules data with the one-shot LLM category encoding. Used
# to verify dataset coverage across size bins.
#
# Input:  data/community_rules_data.csv         (main rules dataset)
#         data/one_shot_llm_category_encoding.csv  (GPT category outputs)
# Output: data/deduplicated_oneshot.csv          (merged dataset with bin labels)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import html
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import seaborn as sns
import ast
import random
plt.style.use(r'code\main_stylesheet.mplstyle')
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]
sns.set_palette(PALETTE)

# Load main rules dataset and aggregate one row per instance to get user counts
df= pd.read_csv(r'data\primary\community_rules_data.csv')
# df = pd.read_csv(r'temp2.csv')
print(len(df))
print(df.columns)
df = df.dropna(subset='translated text')
merged_df = df.copy()

grouped_df = merged_df.groupby('Instance Name')


print(merged_df.columns)

grouped_df = merged_df.groupby('Instance Name').agg(
    All_Rules=('translated text', lambda x: ' '.join(x)),  # Concatenate rules
    Total_User_Count=('User Count', 'first'),               # Sum User Count
    Instance_Group=('instance group', 'first')            # Take the first instance group
).reset_index()

print(grouped_df.head())


# Define the bins and labels
bins = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]  
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$", r"$10^{7}$"]

grouped_df['User Count Bin'] = pd.cut(grouped_df['Total_User_Count'], bins=bins, labels=bin_labels, right=False)

from collections import Counter

print(Counter(grouped_df['User Count Bin']))

#show me instances in 10^7 bin 

print(grouped_df[grouped_df['User Count Bin'] == r"$10^{7}$"])