# wayback_rule_categories.py
#
# Runs GPT-4o rule categorization on pre- and post-Elon Wayback snapshots to
# compare how rule topics changed across three time periods: Sep-Oct 2022,
# Jan 2023, and Jan 2024. Groups categorized rules by instance and merges
# results into a single comparison table.
#
# Input:  non_personal_preelon_notnull.csv   (pre-Elon rules, filtered)
#         non_personal_preelon_2022.csv      (pre-Elon user counts)
#         non_personal_postelon_2024(1).csv  (post-Elon 2024 rules)
#         non_personal_postelon_2023(1).csv  (post-Elon 2023 rules)
# Output: rule_category_comparison.csv  (GPT categories per period per instance)

# category counts
"""
Here, we analysis the difference in the categories of a set of rules of an instance across different time periods: preelon vs postelon 2023, postelon 2023 vs postelon 2024
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import openai
from collections import Counter
from functools import reduce
from dotenv import load_dotenv

# Load pre-Elon snapshot (Sep-Oct 2022)
df_1 = pd.read_csv(r'non_personal_preelon_notnull.csv')       # Sept-Oct 2022 (Pre-Elon)

temp_df = pd.read_csv(r'non_personal_preelon_2022.csv')   

print(len(df_1), len(temp_df))
print(df_1.columns)
print(temp_df.columns)
# Only keep instance + user_count from temp_df
temp_df = temp_df[['instance', 'user_count']]

# Merge user_count into df_1 based on instance
df_1 = pd.merge(df_1, temp_df, on='instance', how='left')

print(df_1.head())


df_2 = pd.read_csv(r'non_personal_postelon_2024(1).csv')     # Post-Elon 2024
df_4 = pd.read_csv(r'non_personal_postelon_2023(1).csv')     # Post-Elon 2023

# # print(len(df_1), len(df_2), len(df_3), len(df_4))
print(len(df_1), len(df_2), len(df_4))

# ==============================
# 2. Merge datasets step by step
# ==============================
# df_1 = df_1.drop(columns=['period', 'note'])
df_1 = df_1.rename(columns={'timestamp': 'timestamp_preelon', 'rules': 'rules_preelon', 'user_count': 'user_count_preelon'})

df_2 = df_2.drop(columns=['period', 'note'])
df_2 = df_2.rename(columns={'timestamp': 'timestamp_postelon2024', 'rules': 'rules_postelon2024', 'user_count': 'user_count_postelon2024'})
# Merge df_1 and df_2 first, keep suffixes to avoid column collisions
df = pd.merge(df_1, df_2, on='instance', how='inner', suffixes=('_preelon', '_postelon2024'))
print(len(df))
print(df.columns)

# Merge with df_3 (collection 2024)
# df = pd.merge(df, df_3, on='instance', how='inner', suffixes=('', '_collection2024'))
# print(len(df))
# print(df.columns)

df_4 = df_4.drop(columns=['period', 'note'])
df_4 = df_4.rename(columns={'timestamp': 'timestamp_postelon2023', 'rules': 'rules_postelon2023', 'user_count': 'user_count_postelon2023'})
# Merge with df_4 (postelon 2023)
df = pd.merge(df, df_4, on='instance', how='inner', suffixes=('', '_postelon2023'))
print(len(df))
print(df.columns)


# ==============================
# 4. Data Cleaning
# ==============================

#Clean columns: remove all note columns 
df = df.drop(columns=['period', 'note'])
print(df.columns)

import ast

import ast
import pandas as pd

def parse_rules(rule_str):
    """
    Convert the string representation of rules into a list of rule texts.
    If parsing fails or the value is NaN, return [].
    """
    if pd.isna(rule_str):
        return []
    try:
        rules = ast.literal_eval(rule_str)
        if isinstance(rules, list):
            return [rule.get("text", "").strip() for rule in rules if isinstance(rule, dict)]
        return []
    except (ValueError, SyntaxError):
        return []


for col in ["rules_preelon", "rules_postelon2023", "rules_postelon2024"]:
    df[f"{col}_parsed"] = df[col].dropna().apply(parse_rules)


print(df['rules_postelon2023_parsed'])

# Select the columns you want to normalize
df_preelon = df[['instance', 'rules_preelon_parsed']].copy()
df_postelon2023 = df[['instance', 'rules_postelon2023_parsed']].copy()
df_postelon2024 = df[['instance', 'rules_postelon2024_parsed']].copy()

# Explode (each list element becomes a row)
df_preelon = df_preelon.explode('rules_preelon_parsed').rename(columns={'rules_preelon_parsed': 'rule'})
df_postelon2023 = df_postelon2023.explode('rules_postelon2023_parsed').rename(columns={'rules_postelon2023_parsed': 'rule'})
df_postelon2024 = df_postelon2024.explode('rules_postelon2024_parsed').rename(columns={'rules_postelon2024_parsed': 'rule'})

print(df_preelon.head(10))
print(df_postelon2023.head(10))
print(df_postelon2024.head(10))

load_dotenv()
openai.api_key = os.environ.get('OPEN_AI_KEY')


gpt_model = "gpt-4o-mini"

rule_types = [
       'Advertising & Commercialization', 'Copyright/ Piracy',
       'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
       'Links & Outside Content', 'NSFW',
       'Off-topic/topic specific', 'Dogpiling',
       'Reposting/Crossposting', 'Spam', 'Trolling',
       'Incitement of Violence', 'Mis/Disinformation/Conspiracy',
       'Illegal Content', 'Content Warnings', 
        'Impersonation', 
       'Automated tools', 'Not Applicable']

prompt1 = f"""
# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No harassment, dogpiling or doxing of other."
# A: Doxxing/Personal Info, Harassment

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Be hot like Lily." 
# A: Not Applicable

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No trolling, microaggressions, or harassment. See the Code of Conduct for the full list."
# A: Harassment, Trolling

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No incitement of violence or promotion of violent ideologies."
# A: Incitement of Violence 

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Conduct intended to stalk or harass users, impede users from using the service, degrade the performance of the service, or incite others to perform any of the aforementioned actions, is disallowed." 
# A: Copyright/Piracy, Harassment

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Respect the laws of Germany and the Isle of Man, including those related to hate speech and defamation."
# A: Hate Speech, Illegal Content

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Account registrations are monitored and accounts considered spam (including commercial advertising, political campaigning/propaganda, duplicate accounts or impersonating legal entities) will be moderated or removed."
# A: Advertising & Commercialization, Spam

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No racism, sexism, homophobia, transphobia, xenophobia, or casteism."
# A: Hate Speech

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Disallowed content:  commercial messages, all SPAM, and anything that could be considered NSFW (e.g. pornography)."
# A: NSFW, Spam

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No posting or reposting links to otherwise illegal content."
# A: Links & Outside Content, Reposting/Crossposting, Illegal Content

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Sexually explicit or violent media must be marked as sensitive when posting." 
# A: Images

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Banana." 
# A: Not Applicable


Based on the given {rule_types}, select the appropriate categories for the rule:
"""

def assess_rules(df, rule_col="rule", model="gpt-4o-mini"):
    """
    Runs GPT model assessment on each rule in a given dataframe.
    Returns dataframe with a new column 'GPT category'.
    """
    language_assessment = []
    
    for rule in df[rule_col]:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt1},
                {"role": "user", "content": f'Rule: "{rule}"'}
            ],
            temperature=0,
            top_p=0
        )
        language_assessment.append(
            response.choices[0].message.content.strip()
        )
    
    df = df.copy()
    df["GPT category"] = language_assessment
    return df

print(len(df_preelon), len(df_postelon2023), len(df_postelon2024))

# Run for each dataset
df_preelon_assessed = assess_rules(df_preelon, rule_col="rule", model="gpt-4o")
df_postelon2023_assessed = assess_rules(df_postelon2023, rule_col="rule", model="gpt-4o")
df_postelon2024_assessed = assess_rules(df_postelon2024, rule_col="rule", model="gpt-4o")

# Preview
print(df_preelon_assessed.head())
print(df_postelon2023_assessed.head())
print(df_postelon2024_assessed.head())


def group_rules(df, period):
    return (
        df.groupby("instance")
        .agg({
            "rule": list,
            "GPT category": lambda x: x.value_counts().to_dict()
        })
        .rename(columns={
            "rule": f"rule_{period}",
            "GPT category": f"GPT category {period}"
        })
        .reset_index()
    )

# Group each dataset
grouped_preelon = group_rules(df_preelon_assessed, "preelon")
grouped_postelon2023 = group_rules(df_postelon2023_assessed, "postelon2023")
grouped_postelon2024 = group_rules(df_postelon2024_assessed, "postelon2024")

# Merge them all together on "instance"
merged = (
    grouped_preelon
    .merge(grouped_postelon2023, on="instance", how="outer")
    .merge(grouped_postelon2024, on="instance", how="outer")
)

print(merged.head())
merged.to_csv("rule_category_comparison.csv", index=False)
