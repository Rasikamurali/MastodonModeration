# negative_binomial_regression.py
#
# Fits count and proportion regression models for lexical features:
# - Negative Binomial for count DVs (word count, rule count, unique topic count)
#   to handle overdispersion beyond what OLS or Poisson assumes.
# - Beta regression for TTR (bounded 0-1), with Japanese instances excluded
#   because space-based TTR is invalid for logographic scripts.
# Runs all six model specifications (user count, federation, age, and combos).
#
# Input:  data/regression_data_lexicalfeature_fed_birth.csv  (lexical + federation + birth)
#         data/one_shot_llm_category_encoding.csv            (GPT category encodings)
#         data/instance_topics_translated.csv                (for Japanese instance detection)
# Output: nb_regression_summary.txt    (NB regression results for count DVs)
#         beta_regression_summary.txt  (Beta regression results for TTR)

import ast
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from datetime import datetime, timezone

# Load and preprocess per-instance regression dataset
# ── Data loading & preprocessing (mirrors updated_regression.py) ─────────────

df = pd.read_csv(r'data\regression_data_lexicalfeature_fed_birth.csv')
df = df.rename(columns={'User Count_x': 'User Count'})

# ── Merge topic count (from updated_regression.py) ───────────────────────────

topic_df = pd.read_csv(r'data\one_shot_llm_category_encoding.csv')

def ensure_set_of_words(value):
    try:
        evaluated = ast.literal_eval(value) if isinstance(value, str) else value
        return set(evaluated) if isinstance(evaluated, (list, set)) else set()
    except (ValueError, SyntaxError):
        return set()

topic_df['GPT category set'] = topic_df['GPT category set'].apply(ensure_set_of_words)

topics_per_instance = topic_df.groupby('Instance Name').agg(
    Topics=('GPT category set', lambda x: {cat for s in x for cat in s})
).reset_index()
topics_per_instance['Number of Unique Topics'] = topics_per_instance['Topics'].apply(len)

df = pd.merge(df, topics_per_instance[['Instance Name', 'Number of Unique Topics']], on='Instance Name', how='inner')

# ── Common preprocessing ──────────────────────────────────────────────────────

df['Log_User_Count'] = np.log10(df['User Count'])

df = df[df['federating number'] > 0]
df['Log_fed_num'] = np.log10(df['federating number'])

df = df[~df['birth'].str.contains("Failed to retrieve instance information. Status code: ", na=False)]
df['birth'] = df['birth'].replace({'Unknown error': np.nan, 'Data not available': np.nan})
df['birth'] = pd.to_datetime(df['birth'], errors='coerce', utc=True)
df = df.dropna(subset=['birth'])

df = df.dropna(subset=['word count', 'rule count', 'TTR', 'Fk Score', 'Number of Unique Topics'])

now = datetime.now(timezone.utc)
df['age_in_days'] = (now - df['birth']).dt.days
df['age_in_years'] = df['age_in_days'] / 365.25  # rescaled to avoid Hessian singularity

print(f"Final sample size: {len(df)}")

# ── Predictor sets ───────────────────────────────────────────────────────────

predictor_sets = {
    'model1_user':       df[['Log_User_Count']],
    'model2_fed':        df[['Log_fed_num']],
    'model3_age':        df[['age_in_years']],
    'model4a_user_fed':  df[['Log_User_Count', 'Log_fed_num']],
    'model4b_user_age':  df[['Log_User_Count', 'age_in_years']],
    'model5_full':       df[['Log_User_Count', 'Log_fed_num', 'age_in_years']],
}

# ── Negative Binomial Regression (count DVs) ────────────────────────────────
# word count and rule count are non-negative integer counts; NB handles
# overdispersion that OLS and Poisson do not.

count_dvs = ['word count', 'rule count', 'Number of Unique Topics']

with open("nb_regression_summary.txt", "w") as f:
    f.write("Negative Binomial Regression — Count Dependent Variables\n")
    f.write("=" * 80 + "\n")

for model_name, X_raw in predictor_sets.items():
    X = sm.add_constant(X_raw)
    for dv in count_dvs:
        y = df[dv].astype(int)
        try:
            result = sm.NegativeBinomial(y, X).fit(disp=False)
            print(f"\n[NB] {model_name} -> {dv}")
            print(result.summary())
            with open("nb_regression_summary.txt", "a") as f:
                f.write(f"\n[{model_name}] DV: {dv}\n")
                f.write(result.summary().as_text())
                f.write("\n" + "-" * 80 + "\n")
        except Exception as e:
            print(f"[NB] {model_name} -> {dv} FAILED: {e}")

# ── Beta Regression (TTR) ────────────────────────────────────────────────────

# Japanese (and other logographic) instances are excluded because TTR is computed from space-separated tokens, which is not valid for Japanese text.

def has_japanese(text):
    """Detect hiragana, katakana, or CJK unified ideographs."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text))

# Use instance descriptions from the lang reference file to identify Japanese instances
lang_df = pd.read_csv(r'data\instance_topics_translated.csv')
japanese_instances = set(
    lang_df[
        lang_df['Description'].apply(has_japanese) |
        lang_df['Short'].apply(has_japanese) |
        lang_df['Instance Name'].apply(has_japanese)
    ]['Instance Name']
)

df_ttr = df[~df['Instance Name'].isin(japanese_instances)].copy()
print(f"Japanese instances excluded from TTR model: {len(japanese_instances & set(df['Instance Name']))}")
print(f"TTR sample size after exclusion: {len(df_ttr)}")

n = len(df_ttr)
ttr_raw = df_ttr['TTR'].copy()

if (ttr_raw <= 0).any() or (ttr_raw >= 1).any():
    print("Note: TTR contains 0 or 1 values — applying boundary correction.")
    ttr = (ttr_raw * (n - 1) + 0.5) / n
else:
    ttr = ttr_raw

ttr_predictor_sets = {
    'model1_user':       df_ttr[['Log_User_Count']],
    'model2_fed':        df_ttr[['Log_fed_num']],
    'model3_age':        df_ttr[['age_in_years']],
    'model4a_user_fed':  df_ttr[['Log_User_Count', 'Log_fed_num']],
    'model4b_user_age':  df_ttr[['Log_User_Count', 'age_in_years']],
    'model5_full':       df_ttr[['Log_User_Count', 'Log_fed_num', 'age_in_years']],
}

with open("beta_regression_summary.txt", "w") as f:
    f.write("Beta Regression — TTR (Type-Token Ratio, Japanese instances excluded)\n")
    f.write("=" * 80 + "\n")

for model_name, X_raw in ttr_predictor_sets.items():
    X = sm.add_constant(X_raw)
    try:
        result = BetaModel(ttr, X).fit(disp=False)
        print(f"\n[Beta] {model_name} -> TTR")
        print(result.summary())
        with open("beta_regression_summary.txt", "a") as f:
            f.write(f"\n[{model_name}] DV: TTR\n")
            f.write(result.summary().as_text())
            f.write("\n" + "-" * 80 + "\n")
    except Exception as e:
        print(f"[Beta] {model_name} -> TTR FAILED: {e}")
