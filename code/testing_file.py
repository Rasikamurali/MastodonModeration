import pandas as pd 
import numpy as np 
import os 
from matplotlib import pyplot 
import seaborn as sns 
import json 
import csv 

# Opening JSON file and loading the data
# into the variable data


data = []
with open('mastodoncloud_2023-10-11.json') as f:
    for line in f:
        post = json.loads(line)
        data.append(post)

data_2 = [] 
with open('mastodon.social_2023-10-11.json') as f1: 
    for line in f1:
        post = json.loads(line)
        data_2.append(post)

df = pd.DataFrame(data)
df1 = pd.DataFrame(data_2)
print(df1.head())
print(df1.columns)
print(df1['sensitive'])
df1_content = df1['content']
print(df1_content[27])
#print(df1['content'])
print(df['content'])