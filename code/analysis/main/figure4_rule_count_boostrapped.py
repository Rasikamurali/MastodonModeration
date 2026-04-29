# figure4_rule_count_boostrapped.py
#
# Plots the number of rules per Mastodon instance, grouped by instance size bin
# (log scale). Each bin shows: raw scatter (gray), violin distribution, and
# bootstrapped mean (red) with 95% CI. Also reports Spearman correlation
# between instance size and rule count.
#
# Input:  data/community_rules_data.csv  (translated rules with user counts)
# Output: figures/rule_count_bootstrapped.png, figures/rule_count_bootstrapped.pdf
# Run from project root: python code/analysis/main/figure4_rule_count_boostrapped.py

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import os
import html
import regex as re
from scipy.stats import kruskal, spearmanr
# Create inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

#df= pd.read_csv(r'data\translated_rules_dataset.csv')
df= pd.read_csv(r'data\primary\community_rules_data.csv')
print(len(df))
print(df.columns)
df = df.dropna(subset='translated text')


data = df.copy()

data['Instance Name'] = data['Instance Name'].str.strip().str.lower()
# These instances have anomalous rule counts that are not representative of typical instances
data = data[~data['Instance Name'].isin(['libranet.de', 'venera.social', 'social.outsourcedmath.com'])]
# mastodon.social excluded as the flagship instance — its scale is an outlier that would dominate the largest bin
data = data[~data['Instance Name'].isin(['mastodon.social'])]

# Define the bins and labels
bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$ to $10^{2}$ ", 
            r"$10^{2}$ to $10^{3}$",
            r"$10^{3}$ to $10^{4}$", 
            r"$10^{4}$ to $10^{5}$", 
            r"$10^{5}$ to $10^{6}$ ", 
            r"$10^{6}$ to $10^{7}$"]
# Assign each instance to a user count bin
data['User Count Bin'] = pd.cut(data['User Count'], bins=bins, labels=bin_labels, right=False)


data = data.dropna(subset=['rule count'])
print(data[data['rule count'] > 30])


data = data.dropna(subset=['User Count Bin'])
# Bootstrapping function to compute statistics
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
            elif statistic == 'median':
                statistic_value = sample_df[f'{feature}'].median()
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

# Compute bootstrapped means
bootstrap_df = bootstrap_by_user_count_bin(data, 'rule count', n_iterations=1000, statistic='mean')


# Ensure proper categorical ordering
bootstrap_df['User Count Bin'] = pd.Categorical(bootstrap_df['User Count Bin'], categories=bin_labels, ordered=True)
data['User Count Bin'] = pd.Categorical(data['User Count Bin'], categories=bin_labels, ordered=True)

# Numeric encoding for plotting
data['User Count Bin Code'] = data['User Count Bin'].cat.codes
bootstrap_df['User Count Bin Code'] = bootstrap_df['User Count Bin'].cat.codes

# Print bootstrapped mean and confidence intervals
print("Bootstrapped Mean and 95% Confidence Intervals:")
print(bootstrap_df[['User Count Bin', 'mean', 'lower_bound', 'upper_bound']])



#OVERLAY
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)

# Jitter for better visibility in scatter plot
jitter = np.random.normal(0, 0.2, size=len(data))

# Scatter plot for raw rule counts
sns.scatterplot(ax=ax, x=data['User Count Bin Code'] + jitter, y=data['rule count'], 
                color='gray', alpha=0.6)

# Violin plot (distribution of rule counts per instance size bin)
sns.violinplot(ax=ax, data=data, x="User Count Bin Code", y="rule count", 
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

ax.set_ylim(-2, 55)
ax.set_xlabel('Instance Size Bin')
ax.set_ylabel('Rule Count')
ax.set_xticks(bootstrap_df['User Count Bin Code'])
ax.set_xticklabels(bootstrap_df['User Count Bin'], rotation=15)

legend = plt.legend()
legend.set_visible(False)

# Save the plot
rc_overlay_plot = "rc_overlay_vp_bs"
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{rc_overlay_plot}.png", dpi=300)
plt.savefig(f"figures/{rc_overlay_plot}.pdf", dpi=300)
plt.show()

print(data.columns)
# # Spearman correlation between word count and instance size
spearman_corr, spearman_p = spearmanr(data['User Count'], data['rule count'])
print(f"Spearman Correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")

