# wrongcolor_fig2a.py
#
# Earlier version of the topic count distribution plot (Figure 2A) that used
# incorrect color settings. Kept for reference; the corrected version is
# robustness_rule_deduplicate.py. Uses the deduplicated LLM category encoding
# to plot number of unique topics per instance by user count bin.
#
# Input:  data/deduplicated_oneshot.csv  (LLM category encoding on deduplicated rules)
# Output: figures/violin_plot_topic_count_dist.png/.pdf

#Import required libraries
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import ast
from itertools import chain
from scipy.stats import kruskal, spearmanr
plt.style.use(r'C:\Users\rasik\Documents\Independent Study\code\main_stylesheet.mplstyle')
import ast
from scipy.stats import zscore


# Color-blind friendly palette orange, purple, blue, teal, yellow, pink
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]

# Load deduplicated rule category encoding
updated_df = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\deduplicated_oneshot.csv')
print(updated_df.head())
print(updated_df['GPT category set'])


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


#Calculate number of topics per instance
topics_per_instance = updated_df.groupby('Instance Name').agg(
    Topics=('GPT category set', lambda x: {cat: sum(cat in s for s in x) for s in x for cat in s}),
    User_Count=('User Count', 'first')
).reset_index()

#Adding log bins based on user count 
bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]


# Assign each instance to a user count bin
topics_per_instance['User Count Bin'] = pd.cut(topics_per_instance['User_Count'], bins=bins, labels=bin_labels, right=False)


#Adding new column to indicate number of unique topics per instance 
topics_per_instance['Number of Unique Topics'] = topics_per_instance['Topics'].apply(len)

print(topics_per_instance)

#Plot distributions of number of topics per instance bin
plt.figure(figsize=(10, 6))

sns.violinplot(x='User Count Bin', y='Number of Unique Topics', data=topics_per_instance, inner=None, alpha=0.7)

# Overlay with strip plot for all points
sns.stripplot(x='User Count Bin', y='Number of Unique Topics', data=topics_per_instance, color='black', alpha=0.5, jitter=True, dodge=True)

# Display the plot
plt.title('Violin Plot with All Points Visible')
plt.xlabel('User Count Bin')
plt.ylabel('Topic Count')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()

# Define output filename
violin_plot = "figures/violin_plot_topic_count_dist"

# # Save the plot
plt.savefig(f"{violin_plot}.png", dpi=300)
plt.savefig(f"{violin_plot}.pdf", dpi=300)

# # Show the plot
plt.show()



# Remove any extreme values (e.g., word count ≥ 100 if not already filtered)

newdf = topics_per_instance.copy()
print(newdf.columns)
# Perform Kruskal-Wallis test across user count bins
groups = [newdf[newdf['User_Count'] == bin_label]['Number of Unique Topics'] for bin_label in bin_labels]
kruskal_test = kruskal(*groups)

# Spearman correlation between word count and instance size
spearman_corr, spearman_p = spearmanr(topics_per_instance['User_Count'], topics_per_instance['Number of Unique Topics'])

# Print results
print(f"Kruskal-Wallis Test: H-statistic = {kruskal_test.statistic:.4f}, p-value = {kruskal_test.pvalue:.4e}")
print(f"Spearman Correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")

