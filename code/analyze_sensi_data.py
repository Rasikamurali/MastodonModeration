import numpy as np 
import pandas as pd 
import os 
from matplotlib import pyplot as plt
import seaborn as sns 
import json 
import csv 
from find_sensitive import total_toots


import os

def list_files(directory):
    files = os.listdir(directory)
    return files

#Access directory 
directory = r'C:\Users\rasik\Documents\Independent Study\sensitive_data'
file_names = list_files(directory)
#print("Files in directory:", file_names)

#Accessing the file path of the directory 
def list_files_with_paths(directory):
    files_with_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
    return files_with_paths

files_with_paths = list_files_with_paths(directory)

toots = total_toots(files_with_paths)


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

#Getting instance size from instance_details file 
instance_info = pd.read_csv('instance_details.csv')
print(instance_info['instance'])
instance_sizes = list(instance_info['user count'])
print(instance_sizes)
for i in range(len(instance_sizes)): 
    if instance_sizes[i] == "Data not available": 
        instance_sizes[i] = 0
print(instance_sizes)

values_to_remove = ['38623','11928']
my_list = [x for x in instance_sizes if x not in values_to_remove]
print(my_list)

file_name = [] 
file_date = [] 
for file in files_with_paths: 
    name_and_date = (file.split('\\')[-1].split("_"))
    name = name_and_date[0] + name_and_date[1]
    dates = name_and_date[-1].split('.')[0]
    #print(dates)
    file_name.append(name)
    file_date.append(dates)


senstive_tags= [] 
for file in files_with_paths: 
    df = pd.read_csv(rf'{file}')
    senstive_tags.append(len(df))

fractions = [] 
for i in range(len(toots)): 
    fractions.append(senstive_tags[i]/toots[i])

data = {'instance name':file_name, 
        'instance date': file_date, 
        'senstive_tags': senstive_tags, 
        'fractions': fractions}
sensitive_df = pd.DataFrame(data)
#print(sensitive_df.head())

#Need fraction of senstive tags compared to total toots and that will be compared across days 

#Creating visualizations to observe the trend for each instance across the dates 
#Step 1: get data for each instance 
avg_tags =[]
names = list(np.unique(file_name))
for name in names:
    print(name)
    filtered_rows = sensitive_df[sensitive_df['instance name']==name]
    #print(filtered_rows.head())
    #Now we create graphs for each instance 
    dates = filtered_rows['instance date']
    tags = filtered_rows['senstive_tags']
    # plt.plot(dates, tags)
    # plt.xticks(rotation = 90)
    # plt.xlabel("Dates")
    # plt.ylabel("Tags")
    # plt.title(f'Senstive tag trend for {name}')
    # plt.show()

    plt.plot(dates, filtered_rows['fractions'])
    plt.xticks(rotation = 90)
    plt.xlabel("Dates")
    plt.ylabel("Ratio of tags to total toots")
    plt.title(f'Senstive tag ratio trend for {name}')
    plt.show()

    avg_tags.append(np.mean(filtered_rows['senstive_tags']))

plt.plot(my_list, avg_tags, marker = 'o')
# Add names to the circles
for i, txt in enumerate(names):
    plt.annotate(txt, (my_list[i], avg_tags[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()

    #print(filtered_rows)
# testing_file = files_with_paths[1]
# df = pd.read_csv(rf'{testing_file}')
# #print(df.head())
# #print(df.columns)
# content = list(df['content'])
# print(content[0])

# rules_and_about = pd.read_csv('about_rules.csv')
# desc= list(rules_and_about['desc'])
# rules = list(rules_and_about['rules'])
# print(desc[0])
# print(rules[0])