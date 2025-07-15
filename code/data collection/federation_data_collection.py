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
import requests



#df = pd.read_csv(r'Independent Study\data\complete_instance_list.csv')
df = pd.read_csv(r'complete_strat20_translated_rules.csv')
df = df.rename(columns={'instance': 'Instance Name'})
columns = ['Instance Name']
df1 = df[columns]
df1 = df1.drop_duplicates(subset='Instance Name')
#df_sample = df[12000:14000]

federation_list = []
fed_number = []
for index, row in df1.iterrows():
    instance_name = row['Instance Name']

    instance_url = f'https://{instance_name}'
    try:
            response = requests.get(f'{instance_url}/api/v1/instance/peers')
            if response.status_code == 200:
                try:
                    instance_info = response.json()
                    #print(instance_info)
                    fed_number.append(len(instance_info))
                    federation_list.append(instance_info)
                except Exception as json_error:
                # Handle JSON parsing error
                    print(f"JSON parsing error: {json_error}")
                    federation_list.append(['error: json'])
                    fed_number.append(0)
            else:
                print(f"Non-200 response, status code: {response.status_code}")
                federation_list.append([f'error: status {response.status_code}'])
                fed_number.append(0)
                     
    except requests.exceptions.RequestException as req_error: 
        print(f"Request error: {req_error}")
        federation_list.append([f'error: request {req_error}'])
        fed_number.append(0)


print(len(df1))
print(len(federation_list))
df1['federating instances'] = federation_list
df1['federating number'] = fed_number
print(df1.head())
df1.to_csv('federation_data_strat20.csv')