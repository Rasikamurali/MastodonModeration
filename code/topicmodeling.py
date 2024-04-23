import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
import ast 

df = pd.read_csv('all_toot_content.csv')
print(df.head())

# Initialize lists to store data for the new DataFrame
# Initialize lists to store data for the new DataFrame
new_instance_names = []
new_dates = []
new_toots = []

# Iterate through each row of the original DataFrame
for index, row in df.iterrows():
    # Extract instance name, date, and toots for the current row
    instance_name = row['instance name']
    date = row['instance date']
    
    # Parse the string representation of list to a list
    toots_str = row['toot_content']
    toots_list = ast.literal_eval(toots_str)
    
    # Take the first toot from the list of toots
    toot = toots_list[0] if toots_list else None
    
    # Add instance name, date, and the selected toot to the lists
    new_instance_names.append(instance_name)
    new_dates.append(date)
    new_toots.append(toot)

# Create a new DataFrame from the extracted data
new_df = pd.DataFrame({
    'instance_name': new_instance_names,
    'date': new_dates,
    'toot_content': new_toots
})

print(new_df)
new_df.to_csv("new_df.csv")