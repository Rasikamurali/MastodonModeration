import numpy as np 
import pandas as pd 
import os 
from matplotlib import pyplot as plt
import seaborn as sns 
import json 
import csv 
import regex as re 

def list_files(directory):
    files = os.listdir(directory)
    return files

#Access directory 
directory = r'C:\Users\rasik\Documents\Independent Study\sensitive_data'
file_names = list_files(directory)


#Accessing the file path of the directory 
def list_files_with_paths(directory):
    files_with_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    return files_with_paths

files_with_paths = list_files_with_paths(directory)

# toots = total_toots(files_with_paths)


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



# file_name = [] 
# file_date = [] 
# toot_content = []
# for file in files_with_paths: 
#     name_and_date = (file.split('\\')[-1].split("_"))
#     name = name_and_date[0] + name_and_date[1]
#     dates = name_and_date[-1].split('.')[0]
#     #print(dates)
#     file_name.append(name)
#     file_date.append(dates)

#     df = pd.read_csv(rf'{file}')
#     content = df['content']
#     #cleaned_toots = []
#     for c in content: 
#         try:
#         #print(c)
#             toot_content.append(re.sub(r'<.*?>', '', c))
#         except TypeError: 
#             toot_content.append("nan type data")
    
import pandas as pd
import re

file_name = [] 
file_date = [] 
toot_content = []

for file in files_with_paths: 
    name_and_date = (file.split('\\')[-1].split("_"))
    name = name_and_date[0] + name_and_date[1]
    dates = name_and_date[-1].split('.')[0]
    

    df = pd.read_csv(rf'{file}')
    content = df['content']
    file_name.extend([name] * len(df))  # Extend the list with the name for each toot
    file_date.extend([dates] * len(df))  # Extend the list with the date for each toot
    
    for c in content: 
        try:
            toot_content.append(re.sub(r'<.*?>', '', c))
        except TypeError: 
            toot_content.append("nan type data")


# senstive_tags= [] 
# for file in files_with_paths: 
#     df = pd.read_csv(rf'{file}')
#     senstive_tags.append(len(df))



data = {'instance name':file_name, 
        'instance date': file_date, 
        'toot_content': toot_content, }
content_df = pd.DataFrame(data)
print(content_df.head())
content_df.to_csv("content.csv")