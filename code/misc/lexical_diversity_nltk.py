# lexical_diversity_nltk.py
#
# Computes a POS-filtered lexical diversity score per instance: the ratio of
# unique to total informational words (nouns, verbs, adjectives, pronouns).
# Plots diversity vs. instance size with violin + scatter + bootstrapped mean.
# Uses NLTK for tokenization and POS tagging.
#
# Input:  data/community_rules_data.csv  (main rules dataset)
# Output: ld_nltk.png, ld_nltk.pdf  (lexical diversity plot)
#         printed Spearman correlation between user count and diversity

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
plt.style.use(r'code\main_stylesheet.mplstyle')
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]
sns.set_palette(PALETTE)


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


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


def informational_lexical_diversity(text):
    # Tokenize text
    tokens = word_tokenize(text)

    # POS tagging
    pos_tags = pos_tag(tokens)

    # POS categories we care about
    target_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS',   # Nouns
                       'PRP', 'PRP$',                 # Pronouns
                       'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                       'JJ', 'JJR', 'JJS'}            # Adjectives

    # Filter tokens based on POS
    informational_words = [word.lower() for word, pos in pos_tags if pos in target_pos_tags]

    if len(informational_words) == 0:
        return 0  # Avoid division by zero

    # Calculate lexical diversity: unique / total
    diversity_score = len(set(informational_words)) / len(informational_words)

    return diversity_score

# Example usage
data = pd.read_csv(r'data/primary/community_rules_data.csv')

print(data.columns)
data = data.dropna(subset=['translated text']) 
df = data.groupby('Instance Name').agg(
    All_Rules=('translated text', lambda x: ' '.join(x)),  # Concatenate rules
    Total_User_Count=('User Count', 'first'),               # Sum User Count
    Instance_Group=('instance group', 'first')            # Take the first instance group
).reset_index()


text = df['All_Rules'].tolist()  # Get the translated text column and drop NaN values

results = [] 
for t in text: 
    diversity = informational_lexical_diversity(t)
    results.append(diversity)

df['lexical_diversity'] = results

print(df[['All_Rules', 'lexical_diversity']].head())

bins = [1, 10, 100, 1000, 10000, 100000, 1000000 ]  
bin_labels = [r"$10^{1}$ to $10^{2}$ ", 
              r"$10^{2}$ to $10^{3}$", 
              r"$10^{3}$ to $10^{4}$", 
              r"$10^{4}$ to $10^{5}$", 
              r"$10^{5}$ to $10^{6}$ ", 
              r"$10^{6}$ to $10^{7}$"]
# Assign each instance to a user count bin
df['User Count Bin'] = pd.cut(df['Total_User_Count'], bins=bins, labels=bin_labels, right=False)
df = df.dropna(subset=['User Count Bin'])

# plot scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['User Count Bin'], y=df['lexical_diversity'])
plt.show()

#remove outliers without function 
# df = df[df['Lexical_Diversity'] <=40]

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
bootstrap_df = bootstrap_by_user_count_bin(df, 'lexical_diversity', n_iterations=1000, statistic='mean')

# Ensure 'User Count Bin' is a categorical variable with the correct order
bootstrap_df['User Count Bin'] = pd.Categorical(bootstrap_df['User Count Bin'], categories=bin_labels, ordered=True)


df['User Count Bin Code'] = df['User Count Bin'].astype('category').cat.codes
bootstrap_df['User Count Bin Code'] = bootstrap_df['User Count Bin'].astype('category').cat.codes

# max_bin_code = reqd_data['User Count Bin Code'].max()
# reqd_data = reqd_data[reqd_data['User Count Bin Code'] != max_bin_code]
# bootstrap_df = bootstrap_df[bootstrap_df['User Count Bin Code'] != max_bin_code]


fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)

# Jitter for better visibility in scatter plot
jitter = np.random.normal(0, 0.2, size=len(df))

sns.scatterplot(ax=ax, x=df['User Count Bin Code'] + jitter, y=df['lexical_diversity'], 
                color = 'gray', alpha=0.6)
sns.violinplot(ax=ax, data=df, x="User Count Bin Code", y="lexical_diversity", 
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

ax.set_ylim(0, df['lexical_diversity'].max()*1.2)
ax.set_xlabel('Instance Size Bin')
ax.set_ylabel('Lexical Diversity')
ax.set_xticks(bootstrap_df['User Count Bin Code'])
ax.set_xticklabels(bootstrap_df['User Count Bin'], rotation=15)

legend = plt.legend()
legend.set_visible(False)

# # Save the plot
lexical_div_nltk = "ld_nltk"
plt.savefig(f"{lexical_div_nltk}.png", dpi=300)
plt.savefig(f"{lexical_div_nltk}.pdf", dpi=300)
plt.show()

from scipy.stats import kruskal, spearmanr

# Spearman correlation between word count and instance size
print(df.columns)
spearman_corr, spearman_p = spearmanr(df['Total_User_Count'], df['lexical_diversity'])
print(f"Spearman Correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")