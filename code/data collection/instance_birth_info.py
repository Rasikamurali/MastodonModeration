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

#Id and token for Instances API to collect instances 
id = '796768437'
id_token = 'ZMaefRB4oUfPcOx892VsAlWuU5sIYlGWXEiw7DVYt0AXDVUsPBSjsIegFol0EBIm0hKz2NWv7I1WKofhysQlWMoXGWHcJ6z6GJlKhcKXD5FqQ537Bvz0A7Bq3wvAOwHN'



def get_rules(instance_df): 
    instance_birth = [] 
    for index, row in instance_df.iterrows():
        instance_name = row['Instance Name']
        print(instance_name)
        instance_url = f'https://{instance_name}'

        try:
            response = requests.get(f'{instance_url}/api/v1/instance')
            if response.status_code == 200:
                try:
                    instance_info = response.json()
                    # Print to check the structure of the response
                    # print(f"Response from {instance_url}: {instance_info}")
                    
                    # Check if the response is a list or a dictionary
                    if isinstance(instance_info, list):
                        print(f"List found for {instance_url}, first item: {instance_info[0]}")
                        instance_info = instance_info[0]  # Access the first item if it's a list
                    
                    # Safely access 'contact_account' and 'created_at'
                    contact_account = instance_info.get('contact_account')
                    if contact_account:
                        births = contact_account.get('created_at', 'Data not available')
                    else:
                        births = 'Data not available'

                    instance_birth.append(births)

                except ValueError:  # Catch JSON decode errors
                    instance_birth.append("Data not available")

            else:
                print(f"Failed to retrieve instance information for {instance_url}. Status code:", response.status_code)
                instance_birth.append(f"Failed to retrieve instance information. Status code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while accessing {instance_url}: {e}")
            instance_birth.append("Unknown error")


    instance_df['birth'] = instance_birth
    
    instance_rule_count_data = {
        "instance" : list(instance_df['Instance Name'])
    }


    print(instance_rule_count_data)
    instance_rule_count_df = pd.DataFrame(instance_rule_count_data)
    instance_rule_count_df.to_csv('instance_count.csv')


    print(instance_df.head())
    output_filename = 'instance_births5.csv'
    instance_df.to_csv(output_filename, index=False)
    return instance_df, output_filename


if __name__ == '__main__': 
    df = pd.read_csv('data/translated_rules_dataset.csv')
    columns = ['Instance Name']
    df1 = df[columns]
    df1 = df1.drop_duplicates(subset='Instance Name')
    df1 = df1[4000:]
    

    df2 = pd.read_csv(r'complete_strat20_translated_rules.csv')
    print(df2.columns)
    df2 = df2.rename(columns={'instance': 'Instance Name'})
    columns = ['Instance Name']
    df3 = df2[columns]
    df3 = df3.drop_duplicates(subset='Instance Name')
    
    # #Get rules 
    rules, rules_csv = get_rules(df1)
    rules, rules_csv = get_rules(df3)
    

