import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
import requests 
import urllib3
import nltk 

import pandas as pd 
import requests 
import urllib3

# Read the DataFrame
df = pd.read_csv('Instances_maxusers_five_fifteenhundred.csv')

instances = df['Instance Name']
instance_rules = []

#random sampling of 100 instances 
sampled_instances = df.sample(n=40)
print(sampled_instances.head())
#Getting all policies
count = 0 

for index, row in sampled_instances.iterrows():
    print(row['Instance Name']) 
    instance_url = f'https://{row['Instance Name']}'  
    try:
        response = requests.get(f'{instance_url}/api/v1/instance')

        if response.status_code == 200:
            try:
                instance_info = response.json()
                instance_rules.append(instance_info.get('rules', 'Data not available'))
            except requests.exceptions.JSONDecodeError: 
                instance_rules.append("Data not available")
            except KeyError: 
                instance_rules.append("Rule not a field")
        else:
            print(f"Failed to retrieve instance information for {instance_url}. Status code:", response.status_code)
            instance_rules.append(f"Failed to retrieve instance information. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while accessing {instance_url}: {e}")
        instance_rules.append("Unknown error")

# Continue with the rest of your code...
print(instance_rules)

sampled_instances['rules'] = instance_rules

cleaned_rules = []
for rule in instance_rules: 
    instance_ruling = []
    for text in rule: 
        try: 
            instance_ruling.append((list(text.values())[-1]))
        except: 
            instance_ruling.append(("data type is a string"))
    cleaned_rules.append(instance_ruling)
print(cleaned_rules)

sampled_instances['cleaned_rules'] = cleaned_rules

print(sampled_instances.head())
sampled_instances.to_csv('five_fifteenhundred_user_rules.csv')