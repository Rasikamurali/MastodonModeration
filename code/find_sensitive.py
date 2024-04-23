import numpy as np 
import pandas as pd 
import os 
from matplotlib import pyplot 
import seaborn as sns 
import json 
import csv 

import os

def list_files(directory):
    files = os.listdir(directory)
    return files

#Access directory 
directory = r'C:\Users\rasik\Documents\Independent Study\data'
file_names = list_files(directory)
#print("Files in directory:", file_names)

#Accessing the file path of the directory 
def list_files_with_paths(directory):
    files_with_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    return files_with_paths

files_with_paths = list_files_with_paths(directory)
#print(files_with_paths)

#Adjusting name 
new_file_names = []
for file in files_with_paths: 
    file_name = file.split('\\')[-1]
    # Find the index of the first occurrence of '.'
    index_of_dot = file_name.find('.')

    # Replace the first '.' with '_'
    if index_of_dot != -1:  # Ensure '.' is found in the string
        modified_string = file_name[:index_of_dot] + '_' + file_name[index_of_dot + 1:]
    else:
        modified_string = file_name
    
    new_file_names.append(modified_string.split('.')[0])

#print(len(files_with_paths), len(new_file_names))
    
#creating function to take total toots for each file to the next file 
def total_toots(file_name): 
    total_toots = []
    for i in range(len(files_with_paths)): 
        df = pd.read_csv(rf'{files_with_paths[i]}')
        total_toots.append(len(df))
    return total_toots

total_toots(files_with_paths)



# #Accessing the file and read the file 
# for i in range(len(files_with_paths)): 
#     df = pd.read_csv(rf'{files_with_paths[i]}')
#     #print(df.head())
#     # Filter rows where the value in a specific column is 'x'
#     filtered_df = df[df['sensitive']]

#     # Select only the desired columns in the filtered DataFrame
#     selected_columns_df = filtered_df[['id', 'content', 'created_at', 'mentions', 'tags', 'favourites_count', 'replies_count']]  # Add the column names you want to select

#     #Print or use selected_columns_df as needed
#     #selected_columns_df.to_csv(f'{new_file_names[i]}.csv')

# #Opening only one file 
# test = files_with_paths[0]

# df = pd.read_csv(rf'{test}')

# #print(df['content'])
# truth = []
# sensitive_tags = list(df['sensitive'])
# for tag in sensitive_tags: 
#     #print(type(tag))
#     if tag is True: 
#         truth.append("Yes")
#         #print('Yes')

# print(len(truth))

