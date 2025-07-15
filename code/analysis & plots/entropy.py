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
import random

# Color-blind friendly palette orange, purple, blue, teal, yellow, pink
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]

sns.set_palette(PALETTE)

## helpers

def patch_violinplot(ax, palette=PALETTE, n=1, alpha=1, multicolor=True):
    """
    Recolor the outlines of violin patches using a palette
    - palette (list of str): color palette for the patches
    - n (int): number of colors to use from the palette
    - multicolor (bool): whether to color the patches differently. If False, use the default color (orange)
    - alpha (float): transparency
    """
    from matplotlib.collections import PolyCollection

    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        if multicolor is False:
            violins[i].set_edgecolor(c="#F18447")
        else:
            colors = sns.color_palette(palette, n_colors=n) * (len(violins) // n)
            violins[i].set_edgecolor(colors[i])
        violins[i].set_alpha(alpha)


def point_violinplot(
    ax, palette=PALETTE, n=1, pointsize=1, edgecolor="white", multicolor=True
):
    """
    Recolor points in the plot based on the violin facecolor
    - palette (list of str): color palette for the patches
    - n (int): number of colors to use from the palette
    - edgecolor (str): point outline color
    - pointsize (int): point size
    - multicolor (bool): whether to color the patches differently. If False, use the default color (orange)
    - alpha (float): transparency
    """
    from matplotlib.collections import PathCollection

    violins = [art for art in ax.get_children() if isinstance(art, PathCollection)]
    for i in range(len(violins)):
        violins[i].set_sizes([pointsize])  # size
        violins[i].set_edgecolor(edgecolor)  # outline
        violins[i].set_linewidth(1.5)
        if multicolor is False:
            violins[i].set_facecolor(c="#F18447")
        else:
            colors = sns.color_palette(palette, n_colors=n) * (len(violins) // n)
            violins[i].set_facecolor(colors[i])


# Read the data from CSV
updated_df = pd.read_csv(r'data\one_shot_llm_category_encoding.csv')
print(updated_df.head())
print(updated_df['GPT category set'])

# Ensure 'GPT category set' contains a set of words
def ensure_set_of_words(value):
    """
    Convert a string representation of a list/set to a Python set of words.
    Handles errors and non-list/set values gracefully.
    """
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

# Calculate number of topics per instance
# This groups by instance and counts how many times each topic appears for that instance
# The result is a dictionary of topic:count for each instance

topics_per_instance = updated_df.groupby('Instance Name').agg(
    Topics=('GPT category set', lambda x: {cat: sum(cat in s for s in x) for s in x for cat in s}),
    User_Count=('User Count', 'first')
).reset_index()

# Adding log bins based on user count
bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$ to $10^{2}$ ", r"$10^{2}$ to $10^{3}$", r"$10^{3}$ to $10^{4}$", r"$10^{4}$ to $10^{5}$", r"$10^{5}$ to $10^{6}$ ", r"$10^{6}$ to $10^{7}$"]

# Assign each instance to a user count bin
topics_per_instance['User Count Bin'] = pd.cut(topics_per_instance['User_Count'], bins=bins, labels=bin_labels, right=False)

# Define rule types (for entropy calculations later)
rule_types = [
    'Advertising & Commercialization', 'Copyright/ Piracy',
    'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
    'Links & Outside Content', 'NSFW', 'Off-topic/topic specific', 'Dogpiling',
    'Reposting/Crossposting', 'Spam', 'Trolling', 'Incitement of Violence', 
    'Mis/Disinformation/Conspiracy', 'Illegal Content', 'Content Warnings', 
    'Impersonation', 'Automated tools', 'Not Applicable'
]

# Step 2: Aggregate topic counts by bin (same as original)
def aggregate_topics(bin_df):
    """
    Aggregate topic counts for a group of instances.
    Returns a dictionary of topic:total_count for the group.
    """
    combined_topics = {}
    for topic_dict in bin_df['Topics']:
        for topic, count in topic_dict.items():
            combined_topics[topic] = combined_topics.get(topic, 0) + count
    return combined_topics

# Apply aggregation
topic_counts_per_bin = (
    topics_per_instance.groupby(['Instance Name', 'User_Count', 'User Count Bin'])  # Group by User Count Bin
    .apply(aggregate_topics)
    .reset_index().rename(columns={0: 'Topic Counts'})
)

print(topic_counts_per_bin.head())
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

# Print the first few rows
print(topic_counts_df.head())

# Taxonomy for mapping topics to broad categories
taxonomy = {
    'Automated tools': 'Action', 
    'Harassment': 'Behavior',
    'Doxxing/ Personal Info': 'Behavior',
    'Dogpiling': 'Behavior',
    'Impersonation': 'Behavior',
    'Trolling': 'Behavior',
    'Hate Speech': 'Content',
    'NSFW': 'Content',
    'Content Warnings': 'Content',
    'Illegal Content': 'Content',
    'Incitement of Violence': 'Content',
    'Mis/Disinformation/Conspiracy': 'Content',
    'Images': 'Content',
    'Off-topic/topic specific': 'Content',
    'Links & Outside Content': 'Content',
    'Spam': 'Distribution',
    'Advertising & Commercialization': 'Distribution',
    'Copyright/ Piracy': 'Distribution',
    'Reposting/Crossposting': 'Distribution',
    'User Count Bin': 'User Count Bin'
}

# Function to map topics to broad categories
def map_to_broad_categories(topic_count_dict, taxonomy):
    """
    Map fine-grained topic counts to broad categories using a taxonomy.
    Returns a dictionary of broad_category:count.
    """
    broad_category_counts = {'Action': 0, 'Behavior': 0, 'Content': 0, 'Distribution': 0}
    for topic, count in topic_count_dict.items():
        broad_category = taxonomy.get(topic)
        if broad_category:
            broad_category_counts[broad_category] += count
    return broad_category_counts

# Apply this mapping while keeping both columns
topic_counts_per_bin['Broad Category Counts'] = topic_counts_per_bin['Topic Counts'].apply(
    lambda topic_dict: map_to_broad_categories(topic_dict, taxonomy)
)

# Calculate entropy for broad categories
def calculate_broad_entropy(broad_counts):
    """
    Calculate the entropy (diversity) of a distribution of broad category counts.
    Returns the entropy value.
    """
    total = sum(broad_counts.values())
    if total == 0:
        return 0
    probs = [count / total for count in broad_counts.values() if count > 0]
    return -sum(p * np.log2(p) for p in probs)

# Apply entropy calculation
topic_counts_per_bin['Entropy'] = topic_counts_per_bin['Broad Category Counts'].apply(calculate_broad_entropy)

# Step 3: Plot entropy across user count bins
ordered_bins = [
    r"$10^{1}$ to $10^{2}$ ", r"$10^{2}$ to $10^{3}$", r"$10^{3}$ to $10^{4}$",
    r"$10^{4}$ to $10^{5}$", r"$10^{5}$ to $10^{6}$ ", r"$10^{6}$ to $10^{7}$"
]

topic_counts_per_bin['User Count Bin'] = pd.Categorical(
    topic_counts_per_bin['User Count Bin'],
    categories=ordered_bins,
    ordered=True
)

# Function to calculate entropy based on category counts (broad categories)
def calculate_category_entropy(broad_category_counts):
    """
    Calculate the entropy of a distribution of broad category counts.
    Returns the entropy value.
    """
    total = sum(broad_category_counts.values())
    if total == 0:
        return 0
    # Calculate the probability distribution for each broad category
    probs = [count / total for count in broad_category_counts.values() if count > 0]
    return -np.sum(list(p * np.log2(p) for p in probs))

# Apply category entropy calculation
topic_counts_per_bin['Category_Entropy'] = topic_counts_per_bin['Broad Category Counts'].apply(calculate_category_entropy)

# Step 5: Sum entropy over all categories for total instance entropy
total_entropy = topic_counts_per_bin.groupby('Instance Name')['Category_Entropy'].sum().reset_index(name='Total_Entropy')

# Step 6: Merge total entropy with user count information
merged = total_entropy.merge(
    topic_counts_per_bin[['Instance Name', 'User Count Bin']].drop_duplicates(),
    on='Instance Name',
    how='left'
)

print(merged.columns)

# Function to perform bootstrapping for confidence intervals by user count bin
def bootstrap_by_user_count_bin(df, feature, n_iterations=1000, statistic='mean'):
    """
    Perform bootstrapping to estimate confidence intervals for a feature grouped by user count bin.
    Returns a DataFrame with mean and confidence intervals for each bin.
    """
    user_bins = df['User Count Bin'].unique()
    bootstrap_results = []
    
    for bin_label in user_bins:
        group_df = df[df['User Count Bin'] == bin_label]
        
        # Perform bootstrapping
        bootstrap_statistics = []
        for _ in range(n_iterations):
            sample_df = group_df.sample(n=len(group_df), replace=True, random_state=random.randint(1, 10000))
            if statistic == 'mean':
                statistic_value = sample_df[f'{feature}'].mean()
            elif statistic == 'mean':
                statistic_value = sample_df[f'{feature}'].mean()
            bootstrap_statistics.append(statistic_value)
        
        # Calculate confidence intervals for the group
        lower_bound = np.percentile(bootstrap_statistics, 2.5)
        upper_bound = np.percentile(bootstrap_statistics, 97.5)
        mean_value = np.mean(bootstrap_statistics)
        
        bootstrap_results.append({
            'User Count Bin': bin_label,
            'mean': mean_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    
    return pd.DataFrame(bootstrap_results)

# Get bootstrapped statistics by user count bin
bootstrap_df = bootstrap_by_user_count_bin(merged, 'Total_Entropy', n_iterations=1000, statistic='mean')

# Ensure 'User Count Bin' is a categorical variable with the correct order
bootstrap_df['User Count Bin'] = pd.Categorical(bootstrap_df['User Count Bin'], categories=bin_labels, ordered=True)

# Convert categorical bins to numerical codes
merged['User Count Bin Code'] = merged['User Count Bin'].astype('category').cat.codes
bootstrap_df['User Count Bin Code'] = bootstrap_df['User Count Bin'].astype('category').cat.codes

# Plotting: Entropy (topic diversity) vs. instance size bin
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)

# Jitter for better visibility in scatter plot
jitter = np.random.normal(0, 0.2, size=len(merged))

# Scatter plot of individual instances
sns.scatterplot(ax=ax, x=merged['User Count Bin Code'] + jitter, y=merged['Total_Entropy'], 
                color = 'gray', alpha=0.6)
# Violin plot for distribution in each bin
sns.violinplot(ax=ax, data=merged, x="User Count Bin Code", y="Total_Entropy", 
               inner=None, alpha=0.3)
# Overlay bootstrapped mean points
sns.scatterplot(ax=ax, x=bootstrap_df['User Count Bin Code'], y=bootstrap_df['mean'], 
                color='red', s=20)

# Confidence interval as error bars
ax.errorbar(bootstrap_df['User Count Bin Code'], bootstrap_df['mean'], 
            yerr=[bootstrap_df['mean'] - bootstrap_df['lower_bound'], 
                  bootstrap_df['upper_bound'] - bootstrap_df['mean']],
            fmt='o', color='red', capsize=3)

# Additional plot settings
patch_violinplot(ax, alpha=0.6, multicolor=False)
point_violinplot(ax, pointsize=1, edgecolor="gray", multicolor=False)

ax.set_ylim(-0.5, merged['Total_Entropy'].max()*1.2)
ax.set_xlabel('Instance Size Bin')
ax.set_ylabel('Topic Diversity (Entropy)')
ax.set_xticks(bootstrap_df['User Count Bin Code'])
ax.set_xticklabels(bootstrap_df['User Count Bin'], rotation=15)

legend = plt.legend()
legend.set_visible(False)

# Save the plot
# The plot shows the distribution and mean of topic diversity (entropy) across instance size bins
# with confidence intervals

# Save as PNG and PDF
topicdiv_overlay_plot = "td_overlay"
plt.savefig(f"{topicdiv_overlay_plot}.png", dpi=300)
plt.savefig(f"{topicdiv_overlay_plot}.pdf", dpi=300)
plt.show()

# Statistical analysis: Kruskal-Wallis and Spearman correlation
from scipy.stats import kruskal, spearmanr

print(topic_counts_per_bin.columns)
# Spearman correlation between word count and instance size
spearman_corr, spearman_p = spearmanr(topic_counts_per_bin['User_Count'], topic_counts_per_bin['Entropy'])
print(f"Spearman Correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")