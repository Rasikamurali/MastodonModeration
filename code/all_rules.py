import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
import requests 
import urllib3
#from requests import NameResolutionError 

df = pd.read_csv('Instances_maxusers_withten.csv')
print(len(df))
print(df.head())
print(df.columns)
print(df['User Count'])
instances = df['Instance Name']

instance_rules = []

#random sampling of 100 instances 
# sampled_instances = df.sample(n=15)
# print(sampled_instances.head())
# #Getting all policies
# count = 0 


# for index, row in sampled_instances.iterrows():
#     print(count)
#     count = count + 1
for instance in instances:
    print(instance)

    #print(row['Instance Name']) 
    

    instance_url = f'https://{instance}'  # Replace with the URL of the instance you're interested in
    #print(instance_url)
    # Make a GET request to the instance API
    response = requests.get(f'{instance_url}/api/v1/instance')

# Check if the request was successful
    if response.status_code == 200:
        try:
            print("yep, trying")
            #print(instance)
            instance_info = response.json()
            #print(instance_info)
        # Print instance description
            #print("Instance description:", instance_info['rules'])
            # instance_short_desc.append(instance_info['short_description'])
            # instance_user_count.append(list(instance_info['stats'].values())[0])
            # instance_description.append(instance_info['description'])
            instance_rules.append(instance_info['rules'])
        except requests.exceptions.JSONDecodeError: 
            print("JSON not in expected format")
            # instance_description.append("Data not available")
            instance_rules.append("Data not available")
            # instance_short_desc.append("Data not available")
            #instance_user_count.append("Data not available")
        except KeyError: 
            print("error")
            instance_rules.append("Rule not a field")
        except requests.exceptions.ConnectTimeout: 
            print("timedout")
            instance_rules.append("Timeout")
        except requests.exceptions.Timeout:
            print("error")
            instance_rules.append("Timeout")
        except requests.exceptions.ConnectionError: 
            print("error")
            instance_rules.append("connection error")
        except urllib3.exceptions.NameResolutionError: 
            print("error")
            instance_rules.append("Name Resolution Error")
        except urllib3.exceptions.MaxRetryError: 
            print("error")
            instance_rules.append("Max Retry error")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while accessing {instance_url}: {e}")
            instance_rules.append("Unknown error")
        except requests.exceptions.SSLError: 
            print("error")
            instance_rules.append("nah")
        except urllib3.exceptions.ConnectTimeoutError: 
            print("error")
            instance_rules.append("nah")
        

    else:
        print("Failed to retrieve instance information. Status code:", response.status_code)



cleaned_rules = []
for rule in min_users_policies: 
    instance_ruling = []
    for text in rule: 
        try: 
            instance_ruling.append((list(text.values())[-1]))
        except: 
            instance_ruling.append(("data type is a string"))
    cleaned_rules.append(instance_ruling)
print(cleaned_rules)


print(instance_rules)
sampled_instances['rules'] = instance_rules
print(sampled_instances.head())
sampled_instances.to_csv('min_users_rules.csv')