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

df1 = pd.read_csv(r'sampled_categorized_rules_my_mastodon_gpt4omini_0326_wUserCount.csv')
print(len(df1))
print(df1.head())
GPT_category1 = [] 

#Cleaning the GPT category column to remove unnecessary characters
for row in df1['GPT category']: 
    GPT_category1.append(re.sub(r'A:\s*|[\[\]\'"]', '', row).strip())
    
df1['GPT category'] = GPT_category1
df1 = df1.rename({"instance": "Instance Name"}, axis=1)
print(df1.columns)


# #Removing unnecessary stuff 

# # not_app = df1[df1['GPT category'].isin(['Not Applicable'])]
# # not_app.to_csv('Not_Applicable.csv')
# df1 = df1[~df1['GPT category'].isin(['Not Applicable'])]

# df1 = df1[~df1['Instance Name'].isin(['libranet.de', 'venera.social', 'social.outsourcedmath.com', 'mastodon.social'])]

# #Merging with User Count dataset 
# # df_merge = pd.read_csv(r'data\instance_rule_count_wUserCount.csv')
# df_merge = pd.read_csv(r'data\community_rules_data.csv')

# df_merge_subset = df_merge[['Instance Name', 'User Count', 'rule count']].drop_duplicates()


# merged_inner = pd.merge(df1, df_merge_subset, on='Instance Name', how='inner')
# print(merged_inner.columns)
# print(len(merged_inner))

# columns = ['Instance Name', 'rule', 'instance group', 'translated text', 'GPT category',
#            'User Count', 'rule count']

# new_df = merged_inner[columns]

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

updated_df.to_csv('temp2.csv')
# updated_df1 = pd.read_csv(r'data\llm_category_analysis_data.csv')
# print(updated_df1.columns)

# updated_df2 = pd.read_csv(r'strat20_GPTcategories.csv')
# print(updated_df2.columns)
# updated_df2 = updated_df2.rename(columns={'instance group_y': 'instance group'})

# updated_df = pd.concat([updated_df1, updated_df2], axis=0)

# updated_df.to_csv('temp.csv')
# print(updated_df.head())
# print(updated_df['GPT category set'])


# # Ensure 'GPT category set' contains a set of words
# def ensure_set_of_words(value):
#     try:
#         # Safely evaluate the string to convert it into a Python object
#         evaluated_value = ast.literal_eval(value) if isinstance(value, str) else value
#         # Convert to set if it's not already
#         if isinstance(evaluated_value, (list, set)):
#             return set(evaluated_value)
#         else:
#             # If the value is not a list or set, return an empty set
#             return set()
#     except (ValueError, SyntaxError):
#         # If there's an issue with literal_eval, return an empty set
#         return set()

# # Apply the function to the 'GPT category set' column
# updated_df['GPT category set'] = updated_df['GPT category set'].apply(ensure_set_of_words)
# print(updated_df.columns)

# # Calculate categories per Instance
# categories_per_rule = updated_df.groupby('Instance Name').agg(
#     Total_Categories=('GPT category set', lambda x: sum(len(cat) for cat in x))
# ).reset_index()


# '''
# Get frequency of each category across the entire dataset
# '''
# # Flatten the sets into a single list again for frequency calculation
# all_categories = list(chain.from_iterable(updated_df['GPT category set']))

# # Recalculate category frequency
# category_frequency = pd.Series(all_categories).value_counts().reset_index()
# category_frequency.columns = ['Category', 'Frequency']

# print(category_frequency)

# category_frequency = category_frequency[category_frequency['Category'] != 'Not Applicable']


# # Topic Frequency across platform 
# plt.figure(figsize=(10, 6))
# sns.set_palette(PALETTE)
# sns.barplot(data=category_frequency, x='Category', y='Frequency')
# plt.title('Frequency of Categories')
# plt.xlabel('Category')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Define output filename
# category_freq_plot = "category_frequency_plot"

# # Save the plot
# plt.savefig(f"{category_freq_plot}.png", dpi=300)
# plt.savefig(f"{category_freq_plot}.pdf", dpi=300)

# # Show the plot
# plt.show()


# topics_per_instance = updated_df.groupby('Instance Name').agg(
#     Topics=('GPT category set', lambda x: {cat: sum(cat in s for s in x) for s in x for cat in s}),
#     User_Count=('User Count', 'first')
# ).reset_index()


# # Print the resulting DataFrame
# print(topics_per_instance)

# bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
# bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]


# # Assign each instance to a user count bin
# topics_per_instance['User Count Bin'] = pd.cut(topics_per_instance['User_Count'], bins=bins, labels=bin_labels, right=False)


# topics_per_instance['Topic Count'] = topics_per_instance['Topics'].apply(lambda x: sum(x.values()))
# topics_per_instance['Number of Unique Topics'] = topics_per_instance['Topics'].apply(len)



# #Plot distributions of number of topics per instance bin
# plt.figure(figsize=(10, 6))

# sns.violinplot(x='User Count Bin', y='Number of Unique Topics', data=topics_per_instance, inner=None, alpha=0.7)

# # Overlay with strip plot for all points
# sns.stripplot(x='User Count Bin', y='Number of Unique Topics', data=topics_per_instance, color='black', alpha=0.5, jitter=True, dodge=True)

# # Display the plot
# plt.title('Violin Plot with All Points Visible')
# # Customize the plot
# #plt.title('Box plot of topic distribution across instance bins')
# plt.xlabel('User Count Bin')
# plt.ylabel('Topic Count')
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.tight_layout()

# # # Define output filename
# # violin_plot = "violin_plot_topic_count_dist"

# # # Save the plot
# # plt.savefig(f"{violin_plot}.png", dpi=300)
# # plt.savefig(f"{violin_plot}.pdf", dpi=300)

# # Show the plot
# plt.show()



# # Step 2: Aggregate topic counts by bin (same as original)
# def aggregate_topics(bin_df):
#     combined_topics = {}
#     for topic_dict in bin_df['Topics']:
#         for topic, count in topic_dict.items():
#             combined_topics[topic] = combined_topics.get(topic, 0) + count
#     return combined_topics

# # Apply aggregation
# topic_counts_per_bin = (
#     topics_per_instance.groupby('User Count Bin')
#     .apply(aggregate_topics)
#     .rename('Topic Counts')
# )

# # Step 3: Calculate z-scores within each bin
# def calculate_z_scores(topic_dict):
#     topics = list(topic_dict.keys())
#     counts = list(topic_dict.values())
#     mean = sum(counts) / len(counts)
#     std = (sum((x - mean) ** 2 for x in counts) / len(counts)) ** 0.5  # Standard deviation
#     return {topic: (count - mean) / std if std != 0 else 0 for topic, count in topic_dict.items()}

# # Apply z-score calculation
# z_scores_per_bin = topic_counts_per_bin.apply(calculate_z_scores)

# # Step 4: Convert to DataFrame for plotting
# z_scores_df = pd.DataFrame(z_scores_per_bin.tolist(), index=z_scores_per_bin.index).fillna(0)

# # Step 5: Plot z-scores
# z_scores_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
# plt.title('Z-Scores of Topics Within Each User Count Bin')
# plt.xlabel('User Count Bins')
# plt.ylabel('Z-Score')
# plt.xticks(rotation=45, ha ='right')  # Rotate x-axis labels for clarity
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# # # Define output filename
# # zs_instance_stacked_bar = "zscore_topics"

# # # Save the plot
# # plt.savefig(f"{zs_instance_stacked_bar}.png", dpi=300)
# # plt.savefig(f"{zs_instance_stacked_bar}.pdf", dpi=300)

# plt.show()


# # Step 2: Aggregate topic counts by bin
# def aggregate_topics(bin_df):
#     # Combine all topic dictionaries in the bin
#     combined_topics = {}
#     for topic_dict in bin_df['Topics']:
#         for topic, count in topic_dict.items():
#             combined_topics[topic] = combined_topics.get(topic, 0) + count
#     return combined_topics

# # Apply aggregation
# topic_counts_per_bin = (
#     topics_per_instance.groupby('User Count Bin')
#     .apply(aggregate_topics)
#     .rename('Topic Counts')
# )

# # Step 3: Normalize topic counts by total rules in each bin
# def normalize_topic_counts(topic_dict):
#     total_count = sum(topic_dict.values())
#     return {topic: (count / total_count) * 100 for topic, count in topic_dict.items()}

# topic_percentages_per_bin = topic_counts_per_bin.apply(normalize_topic_counts)

# # Step 4: Convert to DataFrame for plotting
# percentages_df = pd.DataFrame(topic_percentages_per_bin.tolist(), index=topic_percentages_per_bin.index).fillna(0)
# print(percentages_df.head())


# # Step 4.1: Calculate total prevalence per topic and sort topics by prevalence
# total_prevalence = percentages_df.sum(axis=0)  # Sum percentages across all bins for each topic
# sorted_topics = total_prevalence.sort_values(ascending=False).index
# percentages_df = percentages_df[sorted_topics]  # Reorder columns in the DataFrame

# #Step 5: Plot
# percentages_df.T.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Blues')
# plt.title('Percentage of Topics Across User Count Bins')
# plt.xlabel('Topics')
# plt.ylabel('Topic Prevalence (%)')
# plt.xticks(rotation=45, ha='right')
# plt.legend(title='User Count Bin', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# # topics_bar = "topics_stackbar"

# # # Save the plot
# # plt.savefig(f"{topics_bar}.png", dpi=300)
# # plt.savefig(f"{topics_bar}.pdf", dpi=300)
# plt.show()


# # Step 5: Plot with x-axis as user count bins
# percentages_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
# plt.title('Percentage of Topics Across User Count Bins')
# plt.xlabel('User Count Bins')
# plt.ylabel('Percentage (%)')
# plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# user_topic_bars = "usercount_topics_bar"

# # Save the plot
# plt.savefig(f"{user_topic_bars}.png", dpi=300)
# plt.savefig(f"{user_topic_bars}.pdf", dpi=300)
# plt.show()

# # Step 4: Filter to top 7 topics per bin
# def get_top_n_topics(topic_dict, n=5):
#     sorted_topics = sorted(topic_dict.items(), key=lambda x: x[1], reverse=True)[:n]
#     return dict(sorted_topics)

# top_7_topics_per_bin = topic_percentages_per_bin.apply(get_top_n_topics)

# # Step 5: Convert to DataFrame for plotting
# percentages_df = pd.DataFrame(top_7_topics_per_bin.tolist(), index=top_7_topics_per_bin.index).fillna(0)

# # Step 6: Plot with x-axis as user count bins
# percentages_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
# plt.title('Top 5 Topics as Percentage Across User Count Bins')
# plt.xlabel('User Count Bins')
# plt.ylabel('Percentage (%)')
# plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# # top_5_rules = "top_5_byrules"

# # # Save the plot
# # plt.savefig(f"{top_5_rules}.png", dpi=300)
# # plt.savefig(f"{top_5_rules}.pdf", dpi=300)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# plt.show()


# #NORMALIZING BY INSTANCES 

# # Step 2: Calculate total instances per bin
# instances_per_bin = (
#     topics_per_instance.groupby('User Count Bin')['Instance Name']
#     .nunique()
#     .rename('Total Instances')
# )

# def normalize_by_instances(row):
#     total_instances = instances_per_bin[row['User Count Bin']]
#     normalized_topics = {topic: count / total_instances for topic, count in row['Topics'].items()}
#     return normalized_topics

# topics_per_instance['Normalized Topics'] = topics_per_instance.apply(normalize_by_instances, axis=1)

# # Step 4: Aggregate normalized topics for each bin
# def aggregate_normalized_topics(bin_df):
#     combined_topics = {}
#     for topic_dict in bin_df['Normalized Topics']:
#         for topic, count in topic_dict.items():
#             combined_topics[topic] = combined_topics.get(topic, 0) + count
#     return combined_topics

# normalized_topic_counts_per_bin = (
#     topics_per_instance.groupby('User Count Bin')
#     .apply(aggregate_normalized_topics)
#     .rename('Normalized Topic Counts')
# )

# # Step 5: Convert to DataFrame for z-score calculation and plotting
# normalized_df = pd.DataFrame(normalized_topic_counts_per_bin.tolist(), index=normalized_topic_counts_per_bin.index).fillna(0)

# from scipy.stats import rankdata

# def calculate_percentiles(column):
#     # Calculate percentiles (rankdata gives rank; dividing by length converts to percentiles)
#     return rankdata(column, method='average') / len(column) * 100


# # Apply percentiles to each topic (column)
# percentile_df = normalized_df.apply(calculate_percentiles, axis=0)
# print(percentile_df.describe())

# print(percentile_df)


# # Step 2: Plot percentiles
# percentile_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
# plt.title('Percentiles of Normalized Topics Across User Count Bins')
# plt.xlabel('User Count Bins')
# plt.ylabel('Percentile (%)')
# plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)

# # Save the plot
# # percentiles_plot = "percentiles_by_instance"
# # plt.savefig(f"{percentiles_plot}.png", dpi=300)
# # plt.savefig(f"{percentiles_plot}.pdf", dpi=300)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# plt.show()

# normalized_df_positive = normalized_df + 1  # Adding 1 to ensure no zero or negative values

# # Apply the Box-Cox transformation
# transformed_df = normalized_df_positive.apply(lambda x: boxcox(x)[0])

# # Plot original vs transformed data for each topic
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# # Plot original data
# normalized_df.plot(kind='box', ax=axes[0], title='Original Normalized Topics', vert=False)
# axes[0].set_xlabel('Normalized Values')

# # Plot transformed data
# transformed_df.plot(kind='box', ax=axes[1], title='Box-Cox Transformed Topics', vert=False)
# axes[1].set_xlabel('Transformed Values')

# plt.tight_layout()
# plt.show()

# # If you'd like to plot the distribution of a single topic (e.g., 'Hate Speech'):

# # Plot original vs transformed distribution for 'Hate Speech'
# plt.figure(figsize=(12, 6))
# plt.hist(normalized_df['Hate Speech'], bins=10, alpha=0.6, label='Original')
# plt.hist(transformed_df['Hate Speech'], bins=10, alpha=0.6, label='Box-Cox Transformed')
# plt.title('Distribution of "Hate Speech" - Original vs Transformed')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()

# zscore_df = normalized_df.apply(zscore, axis=0).fillna(0)



# # Step 6: Plot z-scores with x-axis as user count bins and y-axis as z-scores
# zscore_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
# plt.title('Z-Scores of Normalized Topics Across User Count Bins')
# plt.xlabel('User Count Bins')
# plt.ylabel('Z-Score')
# plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# # topics_instance = "topics_by_instance"

# # # Save the plot
# # plt.savefig(f"{topics_instance}.png", dpi=300)
# # plt.savefig(f"{topics_instance}.pdf", dpi=300)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# plt.show()

# # Step 5: Filter to top 7 topics per bin based on z-scores
# def get_top_n_topics(topic_dict, n=5):
#     sorted_topics = sorted(topic_dict.items(), key=lambda x: x[1], reverse=True)[:n]
#     return dict(sorted_topics)

# top_7_zscore_topics_per_bin = zscore_df.apply(get_top_n_topics, axis=1)

# # Step 6: Convert to DataFrame for plotting
# percentages_df = pd.DataFrame(top_7_zscore_topics_per_bin.tolist(), index=top_7_zscore_topics_per_bin.index).fillna(0)

# # Step 7: Plot with x-axis as user count bins
# percentages_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
# plt.title('Top 5 Topics by Z-Score Across User Count Bins')
# plt.xlabel('User Count Bins')
# plt.ylabel('Z-Score')
# plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# # top_5_by_instance = "top5_topics_by_instance"

# # # Save the plot
# # plt.savefig(f"{top_5_by_instance}.png", dpi=300)
# # plt.savefig(f"{top_5_by_instance}.pdf", dpi=300)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# plt.show()



# #Heat map
# from sklearn.preprocessing import MultiLabelBinarizer

# # Binarize the topics for heatmap generation
# mlb = MultiLabelBinarizer()
# topic_matrix = mlb.fit_transform(topics_per_instance['Topics'])

# # Create a DataFrame of the topics' co-occurrence
# topic_df = pd.DataFrame(topic_matrix, columns=mlb.classes_)
# topic_df['User Count Bin'] = topics_per_instance['User Count Bin']

# # Create a pivot table of topic co-occurrence by user count bin
# heatmap_data = topic_df.groupby('User Count Bin').mean()

# # Plot the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(heatmap_data, cmap='Greens', annot=True, fmt='.2f', cbar=True)
# plt.title('Heatmap by User Count Bin')

# plt.tight_layout()
# # heatmap = "topic_heatmap"

# # # Save the plot
# # plt.savefig(f"{heatmap}.png", dpi=300)
# # plt.savefig(f"{heatmap}.pdf", dpi=300)
# plt.show()


