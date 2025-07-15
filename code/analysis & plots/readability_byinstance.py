import pandas as pd 
import numpy as np 
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import random
import textstat
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

df= pd.read_csv(r'data\community_rules_data.csv')
print(len(df))
print(df.columns)
df = df.dropna(subset='translated text')

rules= df['translated text']

# Filter rows where word count in 'translated text' is greater than 1
df = df[df['translated text'].str.split().str.len() > 1]

merged_df = df.copy()
merged_df['User Count'] = pd.to_numeric(merged_df['User Count'], errors='coerce')
merged_df['Instance Name'] = merged_df['Instance Name'].str.strip().str.lower()
merged_df = merged_df[~merged_df['Instance Name'].isin(['libranet.de', 'venera.social'])]
len(merged_df)
# Define the bins and labels
bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$ to $10^{2}$ ", 
              r"$10^{2}$ to $10^{3}$", 
              r"$10^{3}$ to $10^{4}$", 
              r"$10^{4}$ to $10^{5}$", 
              r"$10^{5}$ to $10^{6}$ ", 
              r"$10^{6}$ to $10^{7}$"
              ]
# Assign each instance to a user count bin
merged_df['User Count Bin'] = pd.cut(merged_df['User Count'], bins=bins, labels=bin_labels, right=False)

# Drop rows with missing translated text
merged_df = merged_df.dropna(subset='translated text')

merged_df = merged_df.groupby('Instance Name').agg(
    User_Count=('User Count', 'first'),
    rule_set=('translated text', lambda x: ' '.join(
        [rule.strip().rstrip('.') + '.' for rule in x.astype(str)])
    ),
    User_Count_Bin=('User Count Bin', 'first')
).reset_index()
print(len(merged_df))

def fk(text):
    text = str(text)
    try:
        score = textstat.flesch_reading_ease(text)
        return score
    except Exception as e:
        return e

merged_df['Fk Score'] = merged_df['rule_set'].apply(fk)
print(len(merged_df))

df = merged_df.copy()
print(df.columns)

df = df.dropna(subset=['User_Count_Bin'])
def bootstrap_by_user_count_bin(df, feature, n_iterations=1000, statistic='mean'):
    user_bins = df['User_Count_Bin'].unique()
    bootstrap_results = []
    
    for bin_label in user_bins:
        group_df = df[df['User_Count_Bin'] == bin_label]
        
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
            'User_Count_Bin': bin_label,
            'mean': mean_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    
    return pd.DataFrame(bootstrap_results)

# Get bootstrapped statistics by user count bin
bootstrap_df = bootstrap_by_user_count_bin(df, 'Fk Score', n_iterations=1000, statistic='mean')

# Ensure 'User Count Bin' is a categorical variable with the correct order
bootstrap_df['User_Count_Bin'] = pd.Categorical(bootstrap_df['User_Count_Bin'], categories=bin_labels, ordered=True)

df['User Count Bin Code'] = df['User_Count_Bin'].cat.codes
bootstrap_df['User Count Bin Code'] = bootstrap_df['User_Count_Bin'].cat.codes

# Print bootstrapped mean and confidence intervals
print("Bootstrapped Mean and 95% Confidence Intervals:")
print(bootstrap_df[['User_Count_Bin', 'mean', 'lower_bound', 'upper_bound']])

# ## SIDE by SIDE
# #Create figure with subplots (1 row, 2 columns)
# fig, axes = plt.subplots(1, 2, figsize=(10, 6))  # 1 row, 2 columns
# plt.subplots_adjust(wspace=0.15)
# # Main plot on the left (axes[0])
# ax_main = axes[0]
# jitter = np.random.normal(0, 0.2, size=len(df))

# sns.scatterplot(ax=ax_main, x=df['User Count Bin Code'] + jitter, y=df['Fk Score'], 
#                 color = 'gray', alpha=0.6)
# sns.violinplot(ax=ax_main, data=df, x="User Count Bin Code", y="Fk Score", 
#                inner=None, alpha=0.3)


# # Additional plot settings
# patch_violinplot(ax_main, alpha=0.3, multicolor=False)
# point_violinplot(ax_main, pointsize=1, edgecolor="gray", multicolor=False)

# ax_main.set_ylim(0, df['Fk Score'].max() * 1.2)  # Extend upper limit
# # yticks = np.linspace(0, 80, num=5)  # Adjust num for more or fewer ticks
# # ax_main.set_yticks(yticks)
# # ax_main.set_yticklabels([f"{tick:.0f}" for tick in yticks])
# ax_main.set_xlabel('Instance Size Bin')
# ax_main.set_ylabel('Flesch-Kincaid Score')
# ax_main.set_xticks(bootstrap_df['User Count Bin Code'])
# ax_main.set_xticklabels(bootstrap_df['User_Count_Bin'], rotation=15)

# # Inset Plot on the right (axes[1])
# ax_inset = axes[1]
# sns.scatterplot(ax=ax_inset, x='User Count Bin Code', y='mean', data=bootstrap_df, 
#                 color='red', s=100, label='Bootstrapped Mean')

# # Confidence Interval as Error Bars

# lower_errors = (bootstrap_df['mean'] - bootstrap_df['lower_bound']).clip(lower=0)
# upper_errors = (bootstrap_df['upper_bound'] - bootstrap_df['mean']).clip(lower=0)

# # ax.errorbar(bootstrap_df['User Count Bin Code'], bootstrap_df['mean'], 
# #             yerr=[bootstrap_df['mean'] - bootstrap_df['lower_bound'], 
# #                   bootstrap_df['upper_bound'] - bootstrap_df['mean']],
# #             fmt='o', color='red', capsize=3)
# ax_inset.errorbar(
#     bootstrap_df['User Count Bin Code'],
#     bootstrap_df['mean'],
#     yerr=[lower_errors, upper_errors],
#     fmt='o', color='red', capsize=3
# )

# # Adjust labels and ticks for inset plot
# ax_inset.get_legend().set_visible(False)
# ax_inset.set_xticks(bootstrap_df['User Count Bin Code'])  
# ax_inset.set_xticklabels(bootstrap_df['User_Count_Bin'], rotation=15)

# # Define tick values from min to max for y-axis of inset
# ax_inset.set_ylim(0, merged_df['Fk Score'].max() * 1.2)  # Extend upper limit
# ax_inset.set_xlabel('Instance Size Bin')
# ax_inset.set_ylabel('')

# plt.subplots_adjust(bottom=0.15)
# # Save the plot
# fk_inset_plot = "fk_raw_vp_bs_side_by_side"

# plt.savefig(f"{fk_inset_plot}.png", dpi=300)
# plt.savefig(f"{fk_inset_plot}.pdf", dpi=300)
# plt.show()


## OVERLAY
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)

# Jitter for better visibility in scatter plot
jitter = np.random.normal(0, 0.2, size=len(df))

sns.scatterplot(ax=ax, x=df['User Count Bin Code'] + jitter, y=df['Fk Score'], 
                color = 'gray', alpha=0.6)
sns.violinplot(ax=ax, data=df, x="User Count Bin Code", y="Fk Score", 
               inner=None, alpha=0.3)
# Overlay bootstrapped mean points
sns.scatterplot(ax=ax, x=bootstrap_df['User Count Bin Code'], y=bootstrap_df['mean'], 
                color='red', s=20)

# Confidence interval as error bars

lower_errors = (bootstrap_df['mean'] - bootstrap_df['lower_bound']).clip(lower=0)
upper_errors = (bootstrap_df['upper_bound'] - bootstrap_df['mean']).clip(lower=0)

# ax.errorbar(bootstrap_df['User Count Bin Code'], bootstrap_df['mean'], 
#             yerr=[bootstrap_df['mean'] - bootstrap_df['lower_bound'], 
#                   bootstrap_df['upper_bound'] - bootstrap_df['mean']],
#             fmt='o', color='red', capsize=3)
ax.errorbar(
    bootstrap_df['User Count Bin Code'],
    bootstrap_df['mean'],
    yerr=[lower_errors, upper_errors],
    fmt='o', color='red', capsize=3
)

# Additional plot settings
patch_violinplot(ax, alpha=0.6, multicolor=False)
point_violinplot(ax, pointsize=1, edgecolor="gray", multicolor=False)

ax.set_ylim(0, df['Fk Score'].max() * 1.2)
ax.set_xlabel('Instance Size Bin')
ax.set_ylabel('Flesch-Kincaid Score')
ax.set_xticks(bootstrap_df['User Count Bin Code'])
ax.set_xticklabels(bootstrap_df['User_Count_Bin'], rotation=15)

legend = plt.legend()
legend.set_visible(False)

plt.subplots_adjust(bottom=0.15)
fk_inset_plot = "fk_overlay_vp_bs"

plt.savefig(f"{fk_inset_plot}.png", dpi=300)
plt.savefig(f"{fk_inset_plot}.pdf", dpi=300)
plt.show()


from scipy.stats import kruskal, spearmanr

# Spearman correlation between word count and instance size
spearman_corr, spearman_p = spearmanr(df['User_Count'], df['Fk Score'])
print(f"Spearman Correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")