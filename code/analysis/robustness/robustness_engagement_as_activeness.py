# robustness_engagement_as_activeness.py
#
# Robustness check: re-plots lexical feature distributions using total weekly
# statuses (post volume) as the activeness proxy instead of user count. Produces
# four plots — word count, rule count, TTR, and FK readability — each binned by
# post volume rather than registered users.
#
# Input:  data/community_rules_data.csv  (translated rules per instance)
#         data/weekly_combined.csv       (total statuses per instance)
# Output: figures/engagement_wc_inset_plot.png/.pdf
#         figures/engagement_rule_inset_plot.png/.pdf
#         figures/engagement_ttr_inset_plot.png/.pdf
#         figures/engagement_fk_inset_plot.png/.pdf

# --- IMPORTS ---
import os
import pandas as pd
import numpy as np
import random
import re
import html
from collections import Counter
from matplotlib import pyplot as plt 
import seaborn as sns 
from itertools import chain
from scipy.stats import spearmanr
from scipy.stats import kruskal

plt.style.use(r'C:\Users\rasik\Documents\Independent Study\code\main_stylesheet.mplstyle')
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]

sns.set_palette(PALETTE)

# ----------------------
# Helper plotting functions (unchanged)
# ----------------------
def patch_violinplot(ax, palette=PALETTE, n=1, alpha=1, multicolor=True):
    from matplotlib.collections import PolyCollection
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        if multicolor is False:
            violins[i].set_edgecolor(c="#F18447")
        else:
            colors = sns.color_palette(palette, n_colors=n) * (len(violins) // n)
            violins[i].set_edgecolor(colors[i])
        violins[i].set_alpha(alpha)

def point_violinplot(ax, palette=PALETTE, n=1, pointsize=1, edgecolor="white", multicolor=True):
    from matplotlib.collections import PathCollection
    violins = [art for art in ax.get_children() if isinstance(art, PathCollection)]
    for i in range(len(violins)):
        violins[i].set_sizes([pointsize])
        violins[i].set_edgecolor(edgecolor)
        violins[i].set_linewidth(1.5)
        if multicolor is False:
            violins[i].set_facecolor(c="#F18447")
        else:
            colors = sns.color_palette(palette, n_colors=n) * (len(violins) // n)
            violins[i].set_facecolor(colors[i])

# ----------------------
# Load data
# ----------------------
df = pd.read_csv(r"C:\Users\rasik\Documents\Independent Study\data\community_rules_data.csv")
df = df.dropna(subset=["translated text"])
df = df[df["translated text"].str.split().str.len() > 1]

df['Instance Name'] = df['Instance Name'].str.strip().str.lower()
df = df[~df['Instance Name'].isin(['libranet.de', 'venera.social', 'social.outsourcedmath.com'])]
df = df[~df['Instance Name'].isin(['mastodon.social'])]

df_weekly = pd.read_csv(r"C:\Users\rasik\Documents\Independent Study\data\weekly_combined.csv")

# Merge in TOTAL STATUSES instead of user count
merged_df = pd.merge(
    df,
    df_weekly[["Instance Name", "Total Statuses"]],
    on="Instance Name",
    how="left"
)

print(merged_df.columns)
# ----------------------
# Clean text + word count
# ----------------------
def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        text = text.lower()
        text = html.unescape(text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        cleaned_list.append(text)
    return cleaned_list

merged_df["cleaned translated text"] = clean_text(merged_df["translated text"])

def word_count(df):
    counts = []
    for _, row in df.iterrows():
        words = re.findall(r"\b\w+\b", row["cleaned translated text"])
        counts.append(len(words))
    return counts

merged_df["word_count"] = word_count(merged_df)

# Remove mastodon.social if desired
merged_df = merged_df[merged_df["Instance Name"] != "mastodon.social"]
print(merged_df.columns)

# ----------------------
# BINNING BY TOTAL STATUSES (POST COUNT)
# ----------------------
# Example bins — adjust as needed

max_val = merged_df["Total Statuses"].max()

# Highest power of 10 needed
max_power = int(np.ceil(np.log10(max_val)))

# Create bins: [0, 10, 100, 1000, ..., inf]
bins = [0] + [10 ** i for i in range(1, max_power + 1)] 
# Labels are the lower bound of each bin
labels = [10 ** i for i in range(len(bins) - 1)]

merged_df["Post Bin"] = pd.cut(
    merged_df["Total Statuses"],
    bins=bins,
    labels=labels,
    right=False,        # [10,100), etc.
    include_lowest=True
)


merged_df = merged_df[merged_df["Post Bin"] < 1_000_000]
# ----------------------
# Aggregate: one row per instance
# ----------------------
merged_df_wc = merged_df.groupby("Instance Name").agg(
    Total_Statuses=("Total Statuses", "first"),
    word_count=("word_count", "sum"),
    Post_Bin=("Post Bin", "first")
).reset_index()

merged_df_wc = merged_df_wc[merged_df_wc["word_count"] < 100]

# ----------------------
# Bootstrapping by Post Bin
# ----------------------
def bootstrap_by_bin(df, feature, bin_label="Post_Bin", n_iterations=1000):
    results = []
    bins = df[bin_label].unique()

    for b in bins:
        group = df[df[bin_label] == b]
        stats = []
        for _ in range(n_iterations):
            sample = group.sample(len(group), replace=True)
            stats.append(sample[feature].mean())

        results.append({
            "Post_Bin": b,
            "mean": np.mean(stats),
            "lower": np.percentile(stats, 2.5),
            "upper": np.percentile(stats, 97.5)
        })

    return pd.DataFrame(results)

bootstrap_df = bootstrap_by_bin(merged_df_wc, "word_count")

bootstrap_df["Post_Bin"] = bootstrap_df["Post_Bin"].astype("category")
merged_df_wc["Post_Bin"] = merged_df_wc["Post_Bin"].astype("category")

bootstrap_df["Bin Code"] = bootstrap_df["Post_Bin"].cat.codes
merged_df_wc["Bin Code"] = merged_df_wc["Post_Bin"].cat.codes



# ----------------------
# PLOT: Total Statuses (posts) vs Word Count
# ----------------------
plt.figure(figsize=(10, 6))

jitter = np.random.normal(0, 0.2, size=len(merged_df_wc))

sns.scatterplot(
    x=merged_df_wc["Bin Code"] + jitter,
    y=merged_df_wc["word_count"],
    color="gray",
    alpha=0.6
)

sns.violinplot(
    data=merged_df_wc,
    x="Bin Code",
    y="word_count",
    inner=None,
    alpha=0.3
)

# overlay bootstrap mean + CI
sns.scatterplot(
    x=bootstrap_df["Bin Code"],
    y=bootstrap_df["mean"],
    color="red",
    s=20
)

plt.errorbar(
    bootstrap_df["Bin Code"],
    bootstrap_df["mean"],
    yerr=[bootstrap_df["mean"] - bootstrap_df["lower"],
          bootstrap_df["upper"] - bootstrap_df["mean"]],
    fmt="o",
    color="red"
)

patch_violinplot(plt.gca(), alpha=0.3, multicolor=False)
point_violinplot(plt.gca(), pointsize=1, edgecolor="gray", multicolor=False)

plt.xlabel("Volume of Engagement")
plt.ylabel("Word Count")
#plt.title("Word Count vs Total Statuses (Posts)")

plt.xticks(
    bootstrap_df["Bin Code"],
    bootstrap_df["Post_Bin"]
)

plt.tight_layout()

wc_inset_plot = "engagement_wc_inset_plot"
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{wc_inset_plot}.png", dpi=300)
plt.savefig(f"figures/{wc_inset_plot}.pdf", dpi=300)
plt.show()

# ----------------------
# Spearman correlation
# ----------------------
rho, p = spearmanr(merged_df_wc["Total_Statuses"], merged_df_wc["word_count"])
print(f"SPEARMAN: rho={rho:.4f}, p={p:.4e}")



# ----------------------
# Now with rule count 

# ----------------------
# Aggregate: one row per instance
# ----------------------

#Remove rows with NaN rule count and rows with rules 2500 
merged_df_rc = merged_df.copy()

merged_df_rc = merged_df_rc.dropna(subset=["rule count"])

# ----------------------
# Bootstrapping by Post Bin
# ----------------------
def bootstrap_by_bin(df, feature, bin_label="Post Bin", n_iterations=1000):
    results = []
    bins = df[bin_label].unique()

    for b in bins:
        group = df[df[bin_label] == b]
        stats = []
        for _ in range(n_iterations):
            sample = group.sample(len(group), replace=True)
            stats.append(sample[feature].mean())

        results.append({
            "Post Bin": b,
            "mean": np.mean(stats),
            "lower": np.percentile(stats, 2.5),
            "upper": np.percentile(stats, 97.5)
        })

    return pd.DataFrame(results)

bootstrap_df = bootstrap_by_bin(merged_df_rc, "rule count")

bootstrap_df["Post Bin"] = bootstrap_df["Post Bin"].astype("category")
merged_df_rc["Post Bin"] = merged_df_rc["Post Bin"].astype("category")

bootstrap_df["Bin Code"] = bootstrap_df["Post Bin"].cat.codes
merged_df_rc["Bin Code"] = merged_df_rc["Post Bin"].cat.codes

# ----------------------
# PLOT: Total Statuses (posts) vs Word Count
# ----------------------
plt.figure(figsize=(10, 6))

jitter = np.random.normal(0, 0.2, size=len(merged_df_rc))

sns.scatterplot(
    x=merged_df_rc["Bin Code"] + jitter,
    y=merged_df_rc["rule count"],
    color="gray",
    alpha=0.6
)

sns.violinplot(
    data=merged_df_rc,
    x="Bin Code",
    y="rule count",
    inner=None,
    alpha=0.3
)

# overlay bootstrap mean + CI
sns.scatterplot(
    x=bootstrap_df["Bin Code"],
    y=bootstrap_df["mean"],
    color="red",
    s=20
)

plt.errorbar(
    bootstrap_df["Bin Code"],
    bootstrap_df["mean"],
    yerr=[bootstrap_df["mean"] - bootstrap_df["lower"],
          bootstrap_df["upper"] - bootstrap_df["mean"]],
    fmt="o",
    color="red"
)

patch_violinplot(plt.gca(), alpha=0.3, multicolor=False)
point_violinplot(plt.gca(), pointsize=1, edgecolor="gray", multicolor=False)

plt.xlabel("Volume of Engagement")
plt.ylabel("Total Rule Count")
#plt.title("Rule Count vs Total Statuses (Posts)")

plt.xticks(
    bootstrap_df["Bin Code"],
    bootstrap_df["Post Bin"]
)

plt.tight_layout()


rule_inset_plot = "engagement_rule_inset_plot"
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{rule_inset_plot}.png", dpi=300)
plt.savefig(f"figures/{rule_inset_plot}.pdf", dpi=300)
plt.show()

# ----------------------
# Spearman correlation
# ----------------------
rho, p = spearmanr(merged_df_rc["Total Statuses"], merged_df_rc["rule count"])
print(f"SPEARMAN: rho={rho:.4f}, p={p:.4e}")


# ----------------------
# Now we do it for TTR 

def calculate_ttr(text):
    words = text.split()  # Split text into words
    num_unique_words = len(set(words))  # Count unique words (types)
    total_words = len(words)  # Count all words (tokens)
    return num_unique_words / total_words if total_words > 0 else 0


merged_df_ttr = merged_df.groupby('Instance Name').agg(
    All_Rules=('cleaned translated text', lambda x: ' '.join(x)),  # Concatenate rules
    Total_Statuses=("Total Statuses", "first"),               # Sum User Count
    Post_Bin=("Post Bin", "first")            # Take the first instance group
).reset_index()

merged_df_ttr['TTR'] = merged_df_ttr['All_Rules'].apply(calculate_ttr)

# ----------------------
# Bootstrapping by Post Bin
# ----------------------
def bootstrap_by_bin(df, feature, bin_label="Post_Bin", n_iterations=1000):
    results = []
    bins = df[bin_label].unique()

    for b in bins:
        group = df[df[bin_label] == b]
        stats = []
        for _ in range(n_iterations):
            sample = group.sample(len(group), replace=True)
            stats.append(sample[feature].mean())

        results.append({
            "Post_Bin": b,
            "mean": np.mean(stats),
            "lower": np.percentile(stats, 2.5),
            "upper": np.percentile(stats, 97.5)
        })

    return pd.DataFrame(results)

bootstrap_df = bootstrap_by_bin(merged_df_ttr, "TTR")

bootstrap_df["Post_Bin"] = bootstrap_df["Post_Bin"].astype("category")
merged_df_ttr["Post_Bin"] = merged_df_ttr["Post_Bin"].astype("category")

bootstrap_df["Bin Code"] = bootstrap_df["Post_Bin"].cat.codes
merged_df_ttr["Bin Code"] = merged_df_ttr["Post_Bin"].cat.codes

# ----------------------
# PLOT: Total Statuses (posts) vs Word Count
# ----------------------
plt.figure(figsize=(10, 6))

jitter = np.random.normal(0, 0.2, size=len(merged_df_ttr))

sns.scatterplot(
    x=merged_df_ttr["Bin Code"] + jitter,
    y=merged_df_ttr["TTR"],
    color="gray",
    alpha=0.6
)

sns.violinplot(
    data=merged_df_ttr,
    x="Bin Code",
    y="TTR",
    inner=None,
    alpha=0.3
)

# overlay bootstrap mean + CI
sns.scatterplot(
    x=bootstrap_df["Bin Code"],
    y=bootstrap_df["mean"],
    color="red",
    s=20
)

plt.errorbar(
    bootstrap_df["Bin Code"],
    bootstrap_df["mean"],
    yerr=[bootstrap_df["mean"] - bootstrap_df["lower"],
          bootstrap_df["upper"] - bootstrap_df["mean"]],
    fmt="o",
    color="red"
)

patch_violinplot(plt.gca(), alpha=0.3, multicolor=False)
point_violinplot(plt.gca(), pointsize=1, edgecolor="gray", multicolor=False)

plt.xlabel("Volume of Engagement")
plt.ylabel("Type Token Ratio")
# plt.title("TTR vs Total Statuses (Posts)")

plt.xticks(
    bootstrap_df["Bin Code"],
    bootstrap_df["Post_Bin"]
)

plt.tight_layout()

ttr_inset_plot = "engagement_ttr_inset_plot"
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{ttr_inset_plot}.png", dpi=300)
plt.savefig(f"figures/{ttr_inset_plot}.pdf", dpi=300)
plt.show()

# ----------------------
# Spearman correlation
# ----------------------
rho, p = spearmanr(merged_df_ttr["Total_Statuses"], merged_df_ttr["TTR"])
print(f"SPEARMAN: rho={rho:.4f}, p={p:.4e}")


#----------------------
# Readability 

merged_df_fk = (
    merged_df
    .groupby("Instance Name", as_index=False)
    .agg(
        Total_Statuses=("Total Statuses", "first"),
        rule_set=("translated text", lambda x: " ".join(
            rule.strip().rstrip(".") + "." for rule in x.astype(str)
        )),
        Post_Bin=("Post Bin", "first")
    )
)
import textstat


def fk(text):
    text = str(text)
    try:
        score = textstat.flesch_reading_ease(text)
        return score
    except Exception as e:
        return e

merged_df_fk['Fk Score'] = merged_df_fk['rule_set'].apply(fk)


# ----------------------
# Bootstrapping by Post Bin
# ----------------------
def bootstrap_by_bin(df, feature, bin_label="Post_Bin", n_iterations=1000):
    results = []
    bins = df[bin_label].unique()

    for b in bins:
        group = df[df[bin_label] == b]
        stats = []
        for _ in range(n_iterations):
            sample = group.sample(len(group), replace=True)
            stats.append(sample[feature].mean())

        results.append({
            "Post_Bin": b,
            "mean": np.mean(stats),
            "lower": np.percentile(stats, 2.5),
            "upper": np.percentile(stats, 97.5)
        })

    return pd.DataFrame(results)

bootstrap_df = bootstrap_by_bin(merged_df_fk, "Fk Score")

bootstrap_df["Post_Bin"] = bootstrap_df["Post_Bin"].astype("category")
merged_df_fk["Post_Bin"] = merged_df_fk["Post_Bin"].astype("category")

bootstrap_df["Bin Code"] = bootstrap_df["Post_Bin"].cat.codes
merged_df_fk["Bin Code"] = merged_df_fk["Post_Bin"].cat.codes

# ----------------------
# PLOT: Total Statuses (posts) vs Word Count
# ----------------------
plt.figure(figsize=(10, 6))

jitter = np.random.normal(0, 0.2, size=len(merged_df_fk))

sns.scatterplot(
    x=merged_df_fk["Bin Code"] + jitter,
    y=merged_df_fk["Fk Score"],
    color="gray",
    alpha=0.6
)

sns.violinplot(
    data=merged_df_fk,
    x="Bin Code",
    y="Fk Score",
    inner=None,
    alpha=0.3
)

# overlay bootstrap mean + CI
sns.scatterplot(
    x=bootstrap_df["Bin Code"],
    y=bootstrap_df["mean"],
    color="red",
    s=20
)

plt.errorbar(
    bootstrap_df["Bin Code"],
    bootstrap_df["mean"],
    yerr=[bootstrap_df["mean"] - bootstrap_df["lower"],
          bootstrap_df["upper"] - bootstrap_df["mean"]],
    fmt="o",
    color="red"
)

patch_violinplot(plt.gca(), alpha=0.3, multicolor=False)
point_violinplot(plt.gca(), pointsize=1, edgecolor="gray", multicolor=False)

plt.xlabel("Volume of Engagement")
plt.ylabel("Flesch-Kincaid Score")
# plt.title("Fk Score vs Total Statuses (Posts)")

plt.xticks(
    bootstrap_df["Bin Code"],
    bootstrap_df["Post_Bin"]
)

plt.tight_layout()
fk_inset_plot = "engagement_fk_inset_plot"
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/{fk_inset_plot}.png", dpi=300)
plt.savefig(f"figures/{fk_inset_plot}.pdf", dpi=300)
plt.show()


# ----------------------
# Spearman correlation
# ----------------------
rho, p = spearmanr(merged_df_fk["Total_Statuses"], merged_df_fk["Fk Score"])
print(f"SPEARMAN: rho={rho:.4f}, p={p:.4e}")
