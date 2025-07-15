#Import required libraries
import pandas as pd 
import numpy as np 
from collections import Counter
from matplotlib import pyplot as plt 
import seaborn as sns 
import ast
from itertools import chain
plt.style.use(r'code\main_stylesheet.mplstyle')
import ast
from scipy.stats import zscore
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Color-blind friendly palette orange, purple, blue, teal, yellow, pink
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]

sns.set_palette(PALETTE)

updated_df = pd.read_csv(r'data\one_shot_llm_category_encoding.csv')
print(updated_df.columns)
print(len(updated_df))
updated_df = updated_df.rename(columns={'Instance Id_x': 'Instance Id'})
# updated_df2 = pd.read_csv(r'temp2.csv')
# print(updated_df2.columns)
# print(len(updated_df2))
# updated_df2 = updated_df2.rename(columns={'instance group_y': 'instance group', 'rule_y': 'rule', 'translated text_y': 'translated text'})
# updated_df = pd.concat([updated_df1, updated_df2], axis=0)
# print(updated_df.head())
# print(updated_df.columns)
# print(len(updated_df))



# Ensure 'GPT category set' contains a set of words
def ensure_set_of_words(value):
    try:
        # Safely evaluate the string to convert it into a Python object
        evaluated_value = ast.literal_eval(value) if isinstance(value, str) else value
        # Convert to set if it's not already
        if isinstance(evaluated_value, (list, set)):
            return set(evaluated_value)
        else:
            # If the value is not a list or set, return an empty set
            return set()
    except (ValueError, SyntaxError):
        # If there's an issue with literal_eval, return an empty set
        return set()

# Apply the function to the 'GPT category set' column
updated_df['GPT category set'] = updated_df['GPT category set'].apply(ensure_set_of_words)
print(updated_df.columns)
print(len(updated_df))
#Calculate number of topics per instance
topics_per_instance = updated_df.groupby('Instance Name').agg(
    Topics=('GPT category set', lambda x: {cat: sum(cat in s for s in x) for s in x for cat in s}),
    User_Count=('User Count', 'first')
).reset_index()

print(topics_per_instance.columns)

#Adding log bins based on user count 
bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$ to $10^{2}$ ", r"$10^{2}$ to $10^{3}$", r"$10^{3}$ to $10^{4}$", r"$10^{4}$ to $10^{5}$", r"$10^{5}$ to $10^{6}$ ", r"$10^{6}$ to $10^{7}$"]

# Assign each instance to a user count bin
topics_per_instance['User Count Bin'] = pd.cut(topics_per_instance['User_Count'], bins=bins, labels=bin_labels, right=False)
rule_types = [
    'Advertising & Commercialization', 'Copyright/ Piracy',
    'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
    'Links & Outside Content', 'NSFW', 'Off-topic/topic specific', 'Dogpiling',
    'Reposting/Crossposting', 'Spam', 'Trolling', 'Incitement of Violence', 
    'Mis/Disinformation/Conspiracy', 'Illegal Content', 'Content Warnings', 
    'Impersonation', 'Automated tools', 'Not Applicable'
]

# Step 2: Aggregate topic counts by instance
def aggregate_topics(bin_df):
    combined_topics = {}
    for topic_dict in bin_df['Topics']:
        for topic, count in topic_dict.items():
            combined_topics[topic] = combined_topics.get(topic, 0) + count
    return combined_topics


# Apply aggregation
topic_counts_per_bin = (
    topics_per_instance.groupby(['Instance Name', 'User Count Bin'])
    .apply(aggregate_topics)
    .rename('Topic Counts')
).reset_index()

print(topic_counts_per_bin.columns)

# Ensure topic_counts_per_bin is a DataFrame while keeping Instance Name and User Count Bin
topic_counts_df = (
    topic_counts_per_bin
    .set_index(['Instance Name', 'User Count Bin'])['Topic Counts']  # Keep User Count Bin
    .apply(pd.Series)  # Expand topic dictionary into separate columns
    .fillna(0)  # Fill missing values with 0
    .reset_index()  # Restore Instance Name and User Count Bin as columns
)

# Ensure all rule types are present in the DataFrame, adding missing ones with default value 0
for col in rule_types:
    if col not in topic_counts_df.columns:
        topic_counts_df[col] = 0

# Convert all rule type columns to integers
topic_counts_df[rule_types] = topic_counts_df[rule_types].astype(int)
topic_counts_df.reset_index(inplace=True)  # Reset index if needed

#Calculating total rules for topic prevalence denominator
topic_counts_df['Total Rules'] = (topic_counts_df[rule_types] > 0).sum(axis=1)

# Ensure Total Rules is integer
topic_counts_df['Total Rules'] = topic_counts_df['Total Rules'].astype(int)

# Prevent division by zero by replacing 0 with NaN
topic_counts_df['Total Rules'].replace(0, np.nan, inplace=True)

# Calculate topic importance
topic_importance_df = topic_counts_df[rule_types].div(topic_counts_df['Total Rules'], axis=0)

# Fill NaN values (from division by zero cases) with 0
topic_importance_df = topic_importance_df.fillna(0)

# Add back 'Instance Name' and 'User Count Bin' for reference
topic_importance_df = topic_counts_df[['Instance Name', 'User Count Bin']].join(topic_importance_df)

#Making a copy (mostly cause Rasika doesn't want to mess up the original dataframe)
topic_importance_df_1= topic_importance_df.copy()
topic_importance_df_1 = topic_importance_df_1.drop(columns=['Instance Name']) #don't need instance names in final plotting

# Group by 'User Count Bin' and aggregate (mean) the rule types (topic prevalence) for each bin
aggregated_topic_importance = topic_importance_df_1.groupby('User Count Bin').mean()
aggregated_topic_importance = aggregated_topic_importance.drop(columns= ['Not Applicable'])

# Reset the index so User Count Bin is a column
aggregated_topic_importance_reset = aggregated_topic_importance.reset_index()

#PLOTTING
# Convert DataFrame to long format for seaborn
df_long = aggregated_topic_importance_reset.reset_index().melt(id_vars='User Count Bin', var_name='Topic', value_name='Importance')

#column order based on ABCD and topic prevalence in ABCD 
column_order = [
    "Automated tools",
    "Harassment", 
    "Doxxing/ Personal Info",
    "Impersonation", 
    "Dogpiling", 
    "Trolling",
    "Hate Speech",
    "Illegal Content",
    "NSFW",
    "Content Warnings",
    "Incitement of Violence",
    "Mis/Disinformation/Conspiracy",
    "Images", 
    "Off-topic/topic specific",
    "Links & Outside Content",
    "Spam",
    "Copyright/ Piracy",
    "Advertising & Commercialization", 
    "Reposting/Crossposting"
]

# Reorder the DataFrame columns
heatmap_data = aggregated_topic_importance_reset.set_index("User Count Bin")[column_order]

# Plot heatmap
plt.figure(figsize=(14, 6))
ax = sns.heatmap(heatmap_data, cmap='Greens', annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Topic Prevalence'})

# Customizations
plt.xlabel('')
plt.ylabel('Instance Size Bin')
plt.yticks(rotation=0, fontsize = 7)
plt.xticks(rotation=20, ha = 'right', fontsize = 7)


#Save plot
heatmap = "topic_heatmap"
plt.subplots_adjust(left = 0.2, bottom=0.2)
plt.savefig(f"{heatmap}.png", dpi=300)
plt.savefig(f"{heatmap}.pdf", dpi=300)
plt.tight_layout()
plt.show()


