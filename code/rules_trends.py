import pandas as pd 
import numpy as np 
import os 
import requests 
from matplotlib import pyplot as plt 
import requests 
import urllib3
import nltk 
import json
from sklearn.model_selection import train_test_split
import time
import seaborn as sns 
import ast 


rule_counts = pd.read_csv(r'final_count_updated.csv')

bins = ['1 to 5', '6 to 15', '16 to 50', '51 to 150', '151 to 500', '501 to 1500', '1501 to 5000', '5001+']

test = rule_counts[rule_counts['rule count']==20]
print(test['Instance Name'])

# # Cap the outliers
rule_counts['rule count'] = rule_counts['rule count'].clip( upper=rule_counts['rule count'].quantile(0.85))


sns.boxplot(rule_counts, x= rule_counts['instance_group'], y=rule_counts['rule count'], order= bins)
plt.show()

# sns.scatterplot(rule_counts, x= rule_counts['instance group'], y=rule_counts['rule count'])

#print(rule_counts.columns)
# sns.histplot(rule_counts['rule count'])
# plt.show()


# # Function to safely convert strings to lists and count the number of admins
# def count_admins(x):
#     if isinstance(x, str) and x:  # Check if x is a non-empty string
#         try:
#             # Attempt to convert the string to a Python literal
#             return len(ast.literal_eval(x))
#         except (ValueError, SyntaxError):
#             # If conversion fails, assume it's a single name or comma-separated string
#             return len(x)
#     return 0

# print(data['Admin/Mod'])
# admins = list(data['Admin/Mod']) 

# data['Admin_Count'] = data['Admin/Mod'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)

# # data['Admin_Count'] = data['Admin/Mod'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x else 0)
# sns.boxplot(data, x= data['instance_group'], y=data['Admin_Count'], order= bins)
# plt.show()

