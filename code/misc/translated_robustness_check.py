# translated_robustness_check.py
#
# Validates translation quality by computing Flesch-Kincaid (FK) readability
# scores on both the original and translated text of each rule. Reports mean
# and normalized differences to confirm translation does not distort readability.
# Most analysis code is commented out; the active portion computes raw statistics.
#
# Input:  data/community_rules_data.csv  (rules with 'rule' and 'translated text' columns)
# Output: printed mean/std of FK score differences (no file output; save lines commented out)

import pandas as pd
from collections import Counter
import textstat
from langdetect import detect, LangDetectException
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

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



community_rules = pd.read_csv(r'data\community_rules_data.csv')
# community_rules = community_rules[:1000]
print(community_rules.columns)
print(len(community_rules))

community_rules = community_rules.dropna(subset=['rule', 'translated text'])

english_rules = community_rules[community_rules['lang'] == 'English']
non_english_rules = community_rules[community_rules['lang'] != 'English']


print(len(english_rules))
print(len(non_english_rules))

SUPPORTED_LANGUAGES = ['de', 'fr', 'es', 'it', 'nl', 'pt', 'fi', 'sv', 'da', 'no', 'pl', 'ro', 'cs', 'sk']

# Function to calculate Flesch-Kincaid score and return language
def detect_lang_and_calculate_fk(text):
    text = str(text)
    try:
        lang = detect(text)

        if lang in SUPPORTED_LANGUAGES:
            try:
                textstat.set_lang(lang)
                score = textstat.flesch_reading_ease(text)
            except Exception as e:
                return lang, "Error in applying FKES"
            return lang, score
        else:
            return lang, "Unsupported language"

    except LangDetectException:
        return "undetected", float('nan')  # Use 'undetected' for failed detection

# Apply to original and translated text
community_rules[['original_lang', 'original_flesch_kincaid']] = community_rules['rule'].apply(
    lambda x: pd.Series(detect_lang_and_calculate_fk(x))
)

community_rules[['translated_lang', 'translated_flesch_kincaid']] = community_rules['translated text'].apply(
    lambda x: pd.Series(detect_lang_and_calculate_fk(x))
)

# # Save to CSV
# community_rules.to_csv(r'community_rules_wActualLang_and_Legibility.csv', index=False)

print(community_rules[['original_lang', 'translated_lang', 'original_flesch_kincaid', 'translated_flesch_kincaid']].head())

df = community_rules.copy()


df = df[df['original_flesch_kincaid'] != 'Unsupported language']
df = df[df['original_flesch_kincaid'] != 'Error in applying FKES']

df = df[df['translated_flesch_kincaid'] != 'Unsupported language']
df = df[df['translated_flesch_kincaid'] != 'Error in applying FKES']

df['original_flesch_kincaid'] = df['original_flesch_kincaid'].astype(float)
df['translated_flesch_kincaid'] = df['translated_flesch_kincaid'].astype(float)
df['differnce'] = df['translated_flesch_kincaid'] - df['original_flesch_kincaid']
df['translated_flesch_kincaid'] = df['translated_flesch_kincaid'].astype(float)

# Normalize the difference by the maximum absolute FK score observed
max_score = max(df['original_flesch_kincaid'].max(), df['translated_flesch_kincaid'].max())

# Calculate normalized difference
df['normalized_difference'] = df['differnce'] / max_score

# Plot: distribution of Flesch-Kincaid scores per language
languages = df['original_lang'].unique()
print(languages)

print(df['differnce'].mean())
print(df['differnce'].std())


print(df['normalized_difference'].mean())
print(df['normalized_difference'].std())

plt.figure(figsize=(10, 6))

for lang in sorted(languages):
    subset = df[df['original_lang'] == lang]
    sns.kdeplot(
        subset['differnce'],  # I assume you meant 'difference', typo
        label=lang,
        fill=True,
        alpha=0.3  # lighter fill so multiple curves are visible
    )

plt.title('Flesch-Kincaid Difference Distributions by Language')
plt.xlabel('Flesch-Kincaid Score Difference (Original - Translated)')
plt.ylabel('Density')
plt.legend(title='Language')
plt.tight_layout()
plt.savefig('fk_diff_distribution_all_languages.png')

plt.show()

plt.figure(figsize=(10, 6))

# Plot overall distribution (aggregate across all languages)
sns.kdeplot(
    df['differnce'].dropna(),  # Drop NA values to avoid issues
    fill=True,
    alpha=0.5
)

plt.xlabel('Readability Score Difference (Original – Translated)')
plt.ylabel('Density')
plt.tight_layout()

plt.savefig('fk_diff_distribution_aggregate.png')

plt.show()


plt.figure(figsize=(10, 6))

# Plot overall distribution (aggregate across all languages)
sns.kdeplot(
    df['normalized_difference'].dropna(),  # Drop NA values to avoid issues
    fill=True,
    alpha=0.5
)

plt.title('Aggregate Flesch-Kincaid Difference Distribution')
plt.xlabel('Flesch-Kincaid Score Difference (Original - Translated)')
plt.ylabel('Density')
plt.tight_layout()

plt.savefig('fk_normdiff_distribution_aggregate.png')

plt.show()







languages = df['lang'].unique()
print(languages)

for lang in sorted(languages):
    subset = df[df['lang'] == lang]
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(subset['original_flesch_kincaid'], label='Original', fill=True, alpha=0.5, color='blue')
    sns.kdeplot(subset['translated_flesch_kincaid'], label='Translated', fill=True, alpha=0.5, color='orange')

    plt.title(f'Flesch-Kincaid Readability Distribution for {lang}')
    plt.xlabel('Flesch-Kincaid Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fk_distribution_{lang}.png')

    plt.show()

results = []

df = df.dropna(subset= ['original_flesch_kincaid'])
df = df.dropna(subset= ['translated_flesch_kincaid'])

for lang in df['original_lang'].unique():
    print(f"Running t-test for: {lang}")
    
    subset = df[df['original_lang'] == lang]
    original_fk = subset['original_flesch_kincaid']
    translated_fk = subset['translated_flesch_kincaid']
    
    if len(original_fk) > 1:  # Make sure there's enough data
        t_stat, p_value = ttest_rel(original_fk, translated_fk)
        results.append({'language': lang, 't_stat': t_stat, 'p_value': p_value, 'n_samples': len(subset)})
    else:
        results.append({'language': lang, 't_stat': None, 'p_value': None, 'n_samples': len(subset)})

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Create a results DataFrame
results_df = pd.DataFrame(results)
print(results_df)
#results_df.to_csv('language_ttest_results.csv', index=False)



results = []

df = df.dropna(subset= ['original_flesch_kincaid'])
df = df.dropna(subset= ['translated_flesch_kincaid'])

for lang in df['original_lang'].unique():
    
    subset = df[df['original_lang'] == lang]
    original_fk = subset['original_flesch_kincaid']
    translated_fk = subset['translated_flesch_kincaid']

    
    var_diff = subset['differnce'].var()
    var_orig = original_fk.var()
    var_trans = translated_fk.var()

    # Ratio of variance of difference to original
    ratio_to_orig = var_diff / var_orig

    # Ratio of variance of difference to translated
    ratio_to_trans = var_diff / var_trans

    print(f"Variance Ratio (Difference / Original): {ratio_to_orig:.4f}")
    print(f"Variance Ratio (Difference / Translated): {ratio_to_trans:.4f}")
    results.append({'language': lang, 'var_og_ratio': ratio_to_orig})
    results.append({'languge': lang,  'var_trans_ratio': ratio_to_trans})
    

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

from scipy import stats

results = []

for lang in df['original_lang'].unique():
    subset = df[df['original_lang'] == lang]
    diff = subset['normalized_difference'].dropna()
    
    if len(diff) < 5:  # Avoid very small samples
        continue
    
    t_statistic, p_value = stats.ttest_1samp(diff, popmean=0)
    one_sided_p = p_value / 2 if t_statistic < 0 else 1.0  # Only consider p if in right direction
    
    results.append({
        'language': lang,
        'n': len(diff),
        'mean_diff': diff.mean(),
        't_stat': t_statistic,
        'p_value (one-sided)': one_sided_p
    })

# Create DataFrame for results
import pandas as pd
lang_ttest_df = pd.DataFrame(results).sort_values(by='p_value (one-sided)')
print(lang_ttest_df)

results = []

for lang in df['original_lang'].unique():
    subset = df[df['original_lang'] == lang]
    diff = subset['differnce'].dropna()
    
    if len(diff) < 5:  # Avoid very small samples
        continue
    
    t_statistic, p_value = stats.ttest_1samp(diff, popmean=0)
    one_sided_p = p_value / 2 if t_statistic < 0 else 1.0  # Only consider p if in right direction
    
    results.append({
        'language': lang,
        'n': len(diff),
        'mean_diff': diff.mean(),
        't_stat': t_statistic,
        'p_value (one-sided)': one_sided_p
    })

lang_ttest_df = pd.DataFrame(results).sort_values(by='p_value (one-sided)')
print(lang_ttest_df)




