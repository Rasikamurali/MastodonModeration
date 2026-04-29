# lame.py
#
# Diagnostic script: counts instances per user count bin from the community
# rules data, and merges the LLM category encoding with user count bin labels.
# Kept as a quick sanity-check reference; the final version of this logic
# lives in counting_instance.py.
#
# Input:  data/community_rules_data.csv           (main rules dataset)
#         data/one_shot_llm_category_encoding.csv  (GPT category outputs)
# Output: data/deduplicated_oneshot.csv            (merged with bin labels)

import pandas as pd
import numpy as np
import regex as re
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from itertools import chain
from scipy.stats import boxcox
plt.style.use(r'code\main_stylesheet.mplstyle')

from scipy.stats import zscore


# Color-blind friendly palette orange, purple, blue, teal, yellow, pink
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]


# Load GPT category output (older batch version; currently active input below)
# df1 = pd.read_csv(r'final_categorized_rules_my_mastodon_gpt4omini_0106.csv')
# df1= pd.read_csv(r'sampled_categorized_rules_my_mastodon_gpt4omini_0326(1).csv')

df1 = pd.read_csv(r'data\llm categorization\categorized_full_mastodon_rule_set.csv')
print(len(df1))
print(df1.head())
GPT_category1 = [] 

#Cleaning the GPT category column to remove unnecessary characters
for row in df1['GPT category']: 
    GPT_category1.append(re.sub(r'A:\s*|[\[\]\'"]', '', row).strip())
    
df1['GPT category'] = GPT_category1
df1 = df1.rename({"instance": "Instance Name"}, axis=1)
print(df1.columns)

# Load the main rules dataset to get user counts for binning
new_df = df1.copy()

# Ensure the 'GPT category set' column contains sets
new_df['GPT category set'] = new_df['GPT category'].apply(lambda x: set(x.split(', ')) if x else set())

#new_df.to_csv('final_categorized_withusercount.csv')
# Function to count the words in the set
def count_words_in_set(gpt_set):
    return len(gpt_set)  # Simply return the size of the set



'''
Getting the number of topics per rule across the dataset to see the distribution
'''
# Apply the function to the column
new_df['Category Count'] = new_df['GPT category set'].apply(count_words_in_set)
counter_categories = Counter(new_df['Category Count'])



#using the list of rule types to expand the dataset to create a dataset that can be used for creating heatmap and other visualizations
rule_types = [
    'Advertising & Commercialization', 'Copyright/ Piracy',
    'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
    'Links & Outside Content', 'NSFW', 'Off-topic/topic specific', 'Dogpiling',
    'Reposting/Crossposting', 'Spam', 'Trolling', 'Incitement of Violence', 
    'Mis/Disinformation/Conspiracy', 'Illegal Content', 'Content Warnings', 
    'Impersonation', 'Automated tools', 'Not Applicable'
]

# Create a new DataFrame by copying the existing one
updated_df = new_df.copy()

# Add the new columns with default values (e.g., NaN or 0)
for col in rule_types:
    updated_df[col] = 0  # Use 0 as the default value

def update_columns(row):
    for col in rule_types:
        if col in row['GPT category set']:  # Check if the column name is in the set
            row[col] = 1  # Update the column value to 1
    return row

# Apply the function row-wise
updated_df = updated_df.apply(update_columns, axis=1)


# Define the mapping for merging categories
category_mapping = {
    'Doxxing': 'Doxxing/ Personal Info',
    'Doxxing/Personal Info': 'Doxxing/ Personal Info',
    'Copyright/Piracy': 'Copyright/ Piracy',
    'Advertising & Commercial': 'Advertising & Commercialization',
    'Advertising & Commercialization': 'Advertising & Commercialization',  # Added for consistency
    'Advertising': 'Advertising & Commercialization', 
    'A': 'Advertising & Commercialization',  # Fixed typo here as well
    'Advertising & Commericialization': 'Advertising & Commercialization'
}

# Function to replace categories based on the mapping
def replace_categories(categories):
    return {category_mapping.get(cat, cat) for cat in categories}

# Apply the mapping to the 'GPT category set' column
updated_df['GPT category set'] = updated_df['GPT category set'].apply(replace_categories)

# Recreate the columns with updated categories
rule_types = list(set(category_mapping.values()).union(rule_types))  # Update rule_types
for col in rule_types:
    if col not in updated_df.columns:
        updated_df[col] = 0  # Add new columns for merged categories

# Update the columns based on the modified category set
updated_df = updated_df.apply(update_columns, axis=1)

print(updated_df.head())

updated_df.to_csv(r'data\primary\one_shot_llm_category_encoding.csv', index=False)

