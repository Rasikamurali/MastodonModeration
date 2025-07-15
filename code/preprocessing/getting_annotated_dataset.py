import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import html
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import seaborn as sns
import ast 

data = pd.read_csv(r'Independent Study\data\translated_rules_dataset.csv')

print(len(data))
print(data.columns)

# # Reading uncleaned rules
# merged_data = pd.read_csv(r'Independent Study\data\final_rules.csv')
# print(merged_data.columns)

# # Merging data to get User counts
# merged_data = merged_data.dropna(subset=['User Count'])
# merged_df = pd.merge(data, merged_data, on='Instance Name', how='inner')
# print(merged_df.columns)

# # Selected required columns
# columns = ['Instance Name', 
#            'instance group', 'lang', 'translated text', 'Instance Id', 
#            'User Count', 'Description', 'topic', 'federates with', 'Admin/Mod']
# merged_df = merged_df[columns]
# merged_df['User Count'] = pd.to_numeric(merged_df['User Count'], errors='coerce')
# merged_df = merged_df.dropna(subset=['translated text'])


sample_50 = data.sample(n=50)
sample_50.to_csv('sample_50_common_annotation.csv')

sample_100_rasika = data.sample(n=100)
sample_100_bao = data.sample(n=100)

sample_100_rasika.to_csv('Rasika_annotation.csv')
sample_100_bao.to_csv('Bao_annotation.csv')

sample_rasika = pd.concat([sample_50, sample_100_rasika])
sample_bao = pd.concat([sample_50, sample_100_bao])


print(sample_rasika.head(5))
print(sample_bao.head(5))

sample_rasika.to_csv('Rasika_annotation_full.csv')
sample_bao.to_csv('Bao_annotation_full.csv')
