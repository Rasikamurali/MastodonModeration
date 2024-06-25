import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
import seaborn as sns 
from scipy import stats

# # Specify the path to the folder
# folder_path = r"C:\Users\rasik\Documents\Independent Study\data\formality_genre_data"

# # List all files and directories in the folder
# files = os.listdir(folder_path)

# for file in files: 
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path)

#splitting the dfs 
data1 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\1_15_formality_genre.csv')
data2= pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\16_35_formality_genre.csv')
data3 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\36_150_formality_genre.csv')
data4 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\151_500_formality_genre.csv')
data5 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\501_1500_formality_genre.csv')
data6 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\1501_5000_formality_genre.csv')
data7 =pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\formality_genre_data\5001_formality_genre.csv')

pre_150 = pd.concat([data1, data2, data3])
#print(len(pre_150))

post_150 = pd.concat([data4, data5, data6, data7])

def formality(pre_150, post_150): 
    pre_150_fif = [] 
    post_150_fif = []
    for _, row in pre_150.iterrows(): 
        pre_150_fif.append(len(row['formality label']))
    for _, row in post_150.iterrows(): 
        post_150_fif.append(len(row['formality label']))
    return pre_150_fif, post_150_fif

pre_150_fif_list, post_150_fif_list = formality(pre_150, post_150)
t_statistic, p_value = stats.ttest_ind(pre_150_fif_list, post_150_fif_list)
print(t_statistic, p_value)
