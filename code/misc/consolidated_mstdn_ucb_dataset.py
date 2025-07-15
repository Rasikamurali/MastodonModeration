import pandas as pd 
import numpy as np 
import regex as re
from scipy import stats

df = pd.read_csv(r'data\MastodonRules-sample-cleaned-annotated.csv')

# Consolidate categories
def consolidate_categories(row):
    return ', '.join([col for col in df.columns[1:] if row[col] == 1])

df['Consolidated_Categories'] = df.apply(consolidate_categories, axis=1)

print(df)

df.to_csv(r'data\Consolidated_annotated_ucb_dataset.csv')