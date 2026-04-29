# figure_2a_topiccount.py
#
# Plots the number of distinct rule topics per Mastodon instance, grouped by
# instance size bin (log scale). Each bin shows: raw scatter (gray points),
# violin distribution, and bootstrapped mean (red) with 95% CI error bars.
#
# Input:  data/one_shot_llm_category_encoding.csv  (GPT rule topic categories per instance)
# Output: figures/violin_plot_topic_count_dist.png, figures/violin_plot_topic_count_dist.pdf
# Run from project root: python code/analysis/main/figure_2a_topiccount.py

import os
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
# Create inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import kruskal, spearmanr
plt.style.use(r'code\main_stylesheet.mplstyle')
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
    ax, palette=PALETTE, n=1, pointsize=50, edgecolor="white", multicolor=True
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

# regression_wTopicCounts.csv: one row per instance, includes user count and GPT-assigned topic counts
df= pd.read_csv(r'data\regression\regression_lexicalfeature_def_birth_TopicCounts.csv')
print(len(df))
print(df.columns)
df = df.rename(columns={'User Count_x': 'User Count'})
df_new = df.copy()

# Keep only columns needed for this plot
reqd_data = df_new[['Instance Name', 'User Count', 'User Count Bin', 'Topic_Count']]


bins = [1, 10, 100, 1000, 10000, 100000, 1000000 ]  
bin_labels = [r"$10^{1}$ to $10^{2}$ ", 
              r"$10^{2}$ to $10^{3}$", 
              r"$10^{3}$ to $10^{4}$", 
              r"$10^{4}$ to $10^{5}$", 
              r"$10^{5}$ to $10^{6}$ ", 
              r"$10^{6}$ to $10^{7}$"]
# Assign each instance to a user count bin
reqd_data['User Count Bin'] = pd.cut(reqd_data['User Count'], bins=bins, labels=bin_labels, right=False)
reqd_data = reqd_data.dropna(subset=['User Count Bin'])

# Resamples each bin 1000 times (with replacement) to estimate the mean and 95% CI
# without assuming a normal distribution — robust for small or skewed bin samples
def bootstrap_by_user_count_bin(df, feature, n_iterations=1000, statistic='mean'):
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

        # 2.5th and 97.5th percentiles give the 95% confidence interval
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
bootstrap_df = bootstrap_by_user_count_bin(reqd_data, 'Topic_Count', n_iterations=1000, statistic='mean')

# Ensure 'User Count Bin' is a categorical variable with the correct order
bootstrap_df['User Count Bin'] = pd.Categorical(bootstrap_df['User Count Bin'], categories=bin_labels, ordered=True)


reqd_data['User Count Bin Code'] = reqd_data['User Count Bin'].astype('category').cat.codes
bootstrap_df['User Count Bin Code'] = bootstrap_df['User Count Bin'].astype('category').cat.codes

# max_bin_code = reqd_data['User Count Bin Code'].max()
# reqd_data = reqd_data[reqd_data['User Count Bin Code'] != max_bin_code]
# bootstrap_df = bootstrap_df[bootstrap_df['User Count Bin Code'] != max_bin_code]


fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)

# Jitter for better visibility in scatter plot
jitter = np.random.normal(0, 0.2, size=len(reqd_data))

sns.scatterplot(ax=ax, x=reqd_data['User Count Bin Code'] + jitter, y=reqd_data['Topic_Count'], 
                color = 'gray', alpha=0.4)
sns.violinplot(ax=ax, data=reqd_data, x="User Count Bin Code", y="Topic_Count", 
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

ax.set_ylim(0, reqd_data['Topic_Count'].max()*1.05)
ax.set_xlabel('Instance Size Bin')
ax.set_ylabel('Topic Count')
ax.set_xticks(bootstrap_df['User Count Bin Code'])
ax.set_xticklabels(bootstrap_df['User Count Bin'], rotation=15)

legend = plt.legend()
legend.set_visible(False)

# Save the plot
topiccount_overlay_plot = "tc_overlay"
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{topiccount_overlay_plot}.png", dpi=300)
plt.savefig(f"figures/{topiccount_overlay_plot}.pdf", dpi=300)
plt.show()