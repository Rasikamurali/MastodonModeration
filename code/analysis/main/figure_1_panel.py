# figure_1_panel.py
#
# Produces a 2x2 panel figure showing instance size characteristics:
#   (top-left)    CCDF of instance sizes (how many instances have >= X users)
#   (top-right)   Federating number vs. instance size bin
#   (bottom-left) Weekly statuses vs. instance size bin
#   (bottom-right) Weekly logins vs. instance size bin
# Each panel uses fine-grained log-binning + bootstrapped means with 95% CIs.
#
# Input:  data/community_rules_data.csv  (instance user counts)
#         data/federation_data_combined.csv  (federation peer counts per instance)
#         data/weekly_combined.csv          (weekly statuses, logins per instance)
# Output: figures/figure_1_panel.png, figures/figure_1_panel.pdf
# Run from project root: python code/analysis/main/figure_1_panel.py

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import linregress

plt.style.use(r'code\main_stylesheet.mplstyle')
PALETTE = ["#F18447", "#550F6B", "#3863AC", "#209B8A", "#F8D625", "#BC3684"]
sns.set_palette(PALETTE)

# federation_combined: one row per instance with federation (cross-instance connection) data
# weekly_combined: weekly activity metrics (statuses, logins, registrations) per instance
# complete_instance_list: master list of instances with user counts
df = pd.read_csv(r'data\instance meta data\federation_combined.csv')
print(df.columns)
df_weekly_activity = pd.read_csv(r'data\instance meta data\weekly_combined.csv')
df_merge = pd.read_csv(r'data\primary\complete_instance_list.csv')

merged_inner = pd.merge(df_weekly_activity, df_merge, on='Instance Name', how='inner')
print(merged_inner.columns)
merged_inner = merged_inner.rename(columns={'User Count_x': 'User Count', 'Instance': 'Instance Name', 'Total Statuses': 'Statuses',
                              'Total Logins': 'Logins', 'Total Registrations': 'Registrations'})
# Remove top 10% of federating values within each user-count group to suppress extreme outliers
def remove_outliers_iqr(group):
    upper_bound = group['federating number'].quantile(0.90)
    return group[group['federating number'] <= upper_bound]

df = df.groupby('User Count', group_keys=False).apply(remove_outliers_iqr)

# Bin definition
bins = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
bin_labels = [r"$10^{1}$ to $10^{2}$", r"$10^{2}$ to $10^{3}$", r"$10^{3}$ to $10^{4}$",
              r"$10^{4}$ to $10^{5}$", r"$10^{5}$ to $10^{6}$", r"$10^{6}$ to $10^{7}$"]

plot_bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$",  r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]
df['User Count Bin'] = pd.cut(df['User Count'], bins=bins, labels=plot_bin_labels, right=False)
merged_inner['User Count Bin'] = pd.cut(merged_inner['User Count'], bins=bins, labels=plot_bin_labels, right=False)

# Bootstrap function
def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    if len(data) < 2 or data.isnull().any():
        return np.nan, np.nan, np.nan
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(means), lower, upper

# Helper to plot violin + scatter + bootstrapped trend
def plot_distribution(ax, df_raw, xval, yval, bin_edges, bin_labels):
    # === 1. Raw scatterplot ===
    df_clean = df_raw[[xval, yval]].dropna()
    sns.scatterplot(
        data=df_clean,
        x=xval,
        y=yval,
        ax=ax,
        alpha=0.18,
        color='gray',
        s=15,
        edgecolor=None
    )

    # === 2. Bin user count just for summary statistics ===
    df_clean['User Count Bin'] = pd.cut(df_clean[xval], bins=bin_edges, labels=bin_labels, right=False)

    # === 3. Bootstrap mean ± CI per bin ===
    boot = df_clean.groupby('User Count Bin')[yval].apply(lambda x: pd.Series(bootstrap_ci(x))).unstack()
    boot.columns = ['mean', 'ci_lower', 'ci_upper']
    boot = boot.dropna()

    x_locs = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bin_labels)) if plot_bin_labels[i] in boot.index]

 # Convert "$10^{k}$" → 10^k

    # === 4. Plot bootstrapped means + error bars ===
    ax.errorbar(
        x=x_locs,
        y=boot['mean'],
        yerr=[boot['mean'] - boot['ci_lower'], boot['ci_upper'] - boot['mean']],
        fmt='o-', color='red', capsize=1, alpha= 0.8)


    # === 5. Styling ===
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Instance Size")
    ax.set_ylabel(yval)



# Fine-grained log bins (e.g. base 1.01 = ~1% width per bin) give smooth trend lines
# without committing to the coarser decade bins used for the violin plots
def generate_log_bins(series, base=1.01):
    log_min = np.log10(series.min())
    log_max = np.log10(series.max())
    log_bins = np.arange(log_min, log_max, np.log10(base))
    return 10 ** log_bins

# Plotting function
def plot_highres_summary(ax, data, xval, yval, bin_base=5.5):
    data = data[[xval, yval]].dropna()
    sns.scatterplot(data=data, x=xval, y=yval, ax=ax, alpha=0.1, color='gray', s=10, edgecolor=None)

    bins = generate_log_bins(data[xval], base=bin_base)
    data['bin'] = pd.cut(data[xval], bins=bins)

    summary = data.groupby('bin', observed=False)[yval].apply(lambda x: pd.Series(bootstrap_ci(x))).unstack()
    summary.columns = ['mean', 'lo', 'hi']
    summary = summary.dropna()

    bin_centers = [10 ** ((np.log10(interval.left) + np.log10(interval.right)) / 2) for interval in summary.index]
    summary['x'] = bin_centers

    ax.errorbar(summary['x'], summary['mean'],
                yerr=[summary['mean'] - summary['lo'], summary['hi'] - summary['mean']],
                fmt='o-', color='#F18447', capsize=2, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Instance Size")
    if yval == 'federating number':
        ax.set_ylabel("Federating Number")
    else: 
        ax.set_ylabel(yval)



# === Plot Panel ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CCDF — shows the tail of the instance size distribution.
# Each point (x, y) means: y fraction of instances have >= x users.
# Log-log scale reveals the heavy-tailed (power-law-like) distribution.
data_sorted = np.sort(df['User Count'])
ccdf = 1 - np.arange(1, len(data_sorted) + 1) / len(data_sorted)
axs[0, 0].loglog(
    data_sorted, ccdf,
    marker="o", linestyle="none",
    markerfacecolor="none", markeredgecolor="#F18447", markeredgewidth=1
)
axs[0, 0].set_xlabel("Instance Size")
axs[0, 0].set_ylabel("CCDF")
axs[0, 0].grid(True, which="major", linestyle="--", lw=0.5)


# Plot 2, 3, 4 using high-res binning
plot_highres_summary(axs[0, 1], df, 'User Count', 'federating number')
plot_highres_summary(axs[1, 0], merged_inner, 'User Count', 'Statuses')
plot_highres_summary(axs[1, 1], merged_inner, 'User Count', 'Logins')

plt.tight_layout()
# plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/figure_1_panel.png", dpi=300)
plt.savefig("figures/figure_1_panel.pdf", dpi=300)
plt.show()