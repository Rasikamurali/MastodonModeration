"""
robustness_eval.py

This script evaluates the robustness of translation by comparing Flesch-Kincaid readability scores between original and translated rules.
- Loads translation robustness scores from 'data/translation_robustness_scores(2).csv'.
- Cleans the data by removing rows with errors or unsupported languages.
- Computes the difference in Flesch-Kincaid scores between original and translated texts.
- Analyzes and visualizes the distribution of score differences.
"""
import pandas as pd 
import numpy as np 
from collections import Counter 
import matplotlib.pyplot as plt
import seaborn as sns

# Load translation robustness scores
# (Make sure the file path is correct and the file exists)
df = pd.read_csv(r'data\primary\translation_robustness_scores(2).csv')

print(df.columns)
print(len(df))

# Remove rows with missing or error values in relevant columns
df = df.dropna(subset=['rule', 'translated text'])
df = df[df['original_flesch_kincaid'] != 'Unsupported language']
df = df[df['original_flesch_kincaid'] != 'Error in applying FKES']

df = df[df['translated_flesch_kincaid'] != 'Unsupported language']
df = df[df['translated_flesch_kincaid'] != 'Error in applying FKES']

print(len(df))
print(df)

# Convert Flesch-Kincaid columns to float for calculation
df['original_flesch_kincaid'] = df['original_flesch_kincaid'].astype(float)
df['translated_flesch_kincaid'] = df['translated_flesch_kincaid'].astype(float)

# Compute the score difference between original and translated texts
df['Score Difference'] = df['original_flesch_kincaid'] - df['translated_flesch_kincaid']
print(df['Score Difference'].describe())
#df.to_csv(r'translation_robustness_scores_cleaned.csv', index=False)    

df = df[df['Score Difference'] != 0]

# Analyze the direction of score differences
print(Counter(df['Score Difference'] > 0))
print(Counter(df['Score Difference'] < 0))

# DataFrame of rules with score difference of 0
zero_diff = df[df['Score Difference'] == 0]
print(len(zero_diff))
print(zero_diff)
print(zero_diff['Instance Name'].nunique())  
#zero_diff.to_csv(r'zero_diff.csv', index=False)

# Plot the distribution of translated Flesch-Kincaid scores
plt.figure(figsize=(10, 6))
sns.histplot(df['translated_flesch_kincaid'], bins=50, color='black')
plt.xlabel('Flesch-Kincaid Score Difference (Original - Translated)')
plt.ylabel('Frequency')
plt.title('Distribution of Flesch-Kincaid Score Differences')
plt.tight_layout()
plt.show()