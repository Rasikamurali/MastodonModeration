import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
import seaborn as sns 
from scipy import stats
from collections import Counter

# Specify the path to the folder
folder_path = r"C:\Users\rasik\Documents\Independent Study\data\formality_genre_data"

# List all files and directories in the folder
files = os.listdir(folder_path)

instance_group = [] 
for file in files: 
    file_path = os.path.join(folder_path, file)
    group_naming = (file.split('.')[0].split('_'))
    group = group_naming[0] + 'to' + group_naming[1]
    print(group)
    df = pd.read_csv(file_path)
    size = len(df)
    instance_group.extend([group] * size)


#splitting the dfs 
data1 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\1_15_formality_genre.csv')
data2= pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\16_35_formality_genre.csv')
data3 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\36_150_formality_genre.csv')
data4 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\151_500_formality_genre.csv')
data5 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\501_1500_formality_genre.csv')
data6 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\1501_5000_formality_genre.csv')
data7 =pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\5001_formality_genre.csv')

full_data = pd.concat([data1, data2, data3, data4, data5, data6, data7])
full_data['instance group'] = instance_group

full_data.to_csv('complete_data.csv')

print(full_data.head())

print(full_data.columns)
instance_groups = ['1to15', '16to35', '36to150', '151to500', '501to1500', '1501to5000', '5001toformality']

for group in instance_groups: 
    filtered_df = full_data[full_data['instance group']==group]
    #print(filtered_df.head())
    print(group)
    formality_labels = filtered_df['formality label']
    print(Counter(formality_labels))
