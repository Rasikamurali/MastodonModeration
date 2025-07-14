import pandas as pd 
import numpy as np 
from collections import Counter 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'translation_robustness_scores(1).csv')

print(df.columns)
print(len(df))


df = df.dropna(subset=['rule', 'translated text'])
df = df[df['original_flesch_kincaid'] != 'Unsupported language']
df = df[df['original_flesch_kincaid'] != 'Error in applying FKES']

df = df[df['translated_flesch_kincaid'] != 'Unsupported language']
df = df[df['translated_flesch_kincaid'] != 'Error in applying FKES']

print(len(df))
print(df)

df['original_flesch_kincaid'] = df['original_flesch_kincaid'].astype(float)
df['translated_flesch_kincaid'] = df['translated_flesch_kincaid'].astype(float)


df['Score Difference'] = df['original_flesch_kincaid'] - df['translated_flesch_kincaid']
print(df['Score Difference'].describe())
#df.to_csv(r'translation_robustness_scores_cleaned.csv', index=False)    

df = df[df['Score Difference'] != 0]

print(Counter(df['Score Difference'] > 0))
print(Counter(df['Score Difference'] < 0))

#Df of rules with score difference of 0
zero_diff = df[df['Score Difference'] == 0]
print(len(zero_diff))
print(zero_diff)
print(zero_diff['Instance Name'].nunique())  
#zero_diff.to_csv(r'zero_diff.csv', index=False)

#Plot the distribution of score differences
plt.figure(figsize=(10, 6))
sns.histplot(df['translated_flesch_kincaid'], bins=50, color='black')
plt.xlabel('Flesch-Kincaid Score Difference (Original - Translated)')
plt.ylabel('Frequency')
plt.title('Distribution of Flesch-Kincaid Score Differences')
plt.tight_layout()
plt.show()