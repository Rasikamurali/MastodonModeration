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


#Number of instances I want to retrieve 
count = 10000

#Defining function to collect instances 

def collect_instances(min_users, max_users, include_down=False): 
    #df = pd.DataFrame()
    url = f"https://instances.social/api/1.0/instances/list?count={count}&min_users={min_users}&max_users={max_users}"
    
    headers = {
    'Authorization': f'Bearer {id_token}'
    }

    try:
        response = requests.get(url, headers=headers)
        print(response)

        if response.status_code == 200:
            print("Entered pt2!")
            response_data = response.content.decode('utf-8')  # Decode the bytes response to a string
            data = json.loads(response_data)  # Parse the JSON data as a list
            #print(data)
            #print(data)
            # Extract relevant information from the response and create a list of dictionaries
            instance_data = []
            instances = data.get('instances')
            for instance in instances:
                id = instance.get('id')
                name = instance.get('name')
                dead = instance.get('dead')
                user_count = instance.get('users')
                description = instance.get('info').get('full_description')
                short_description = instance.get('info').get('short_description')
                topic =  instance.get('info').get('topic')
                federates_with = instance.get('info').get('federates_with')
                reg = instance.get('dead')
                closed = instance.get('open_registrations')
                admin = instance.get('admin')

                instance_data.append({
                        "Instance Id": id,
                        "Instance Name": name,
                        "Is Dead": dead,
                        "User Count": user_count,
                        "Description": description,
                        "Short": short_description,
                        "topic": topic,
                        "dead": reg, 
                        "federates with": federates_with,
                        "reg": closed, 
                        "Admin/Mod" : admin
                    })

            # Create a DataFrame from the list of dictionaries
            df = pd.DataFrame(instance_data)
            #print(df)
        
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    return df, df.to_csv(f'Instances_{min_users}_{max_users}.csv')


def collect_instances_min(min_users, include_down=False): 
    #df = pd.DataFrame()
    url = f"https://instances.social/api/1.0/instances/list?count={count}&min_users={min_users}"
    
    headers = {
    'Authorization': f'Bearer {id_token}'
    }

    try:
        response = requests.get(url, headers=headers)
        print(response)

        if response.status_code == 200:
            print("Entered pt2!")
            response_data = response.content.decode('utf-8')  # Decode the bytes response to a string
            data = json.loads(response_data)  # Parse the JSON data as a list
            #print(data)
            #print(data)
            # Extract relevant information from the response and create a list of dictionaries
            instance_data = []
            instances = data.get('instances')
            for instance in instances:
                id = instance.get('id')
                name = instance.get('name')
                dead = instance.get('dead')
                user_count = instance.get('users')
                description = instance.get('info').get('full_description')
                short_description = instance.get('info').get('short_description')
                topic =  instance.get('info').get('topic')
                federates_with = instance.get('info').get('federates_with')
                reg = instance.get('dead')
                closed = instance.get('open_registrations')
                admin = instance.get('admin')

                instance_data.append({
                        "Instance Id": id,
                        "Instance Name": name,
                        "Is Dead": dead,
                        "User Count": user_count,
                        "Description": description,
                        "Short": short_description,
                        "topic": topic,
                        "dead": reg, 
                        "federates with": federates_with,
                        "reg": closed, 
                        "Admin/Mod" : admin
                    })

            # Create a DataFrame from the list of dictionaries
            df = pd.DataFrame(instance_data)
            #print(df)
        
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    return df, df.to_csv(f'Instances_{min_users}.csv')

def get_rules(instance_df): 
    instance_rules = [] 
    rule_counts = []

    for index, row in instance_df.iterrows():
        instance_name = row['Instance Name']
        print(instance_name)
        instance_url = f'https://{instance_name}'

        try:
            response = requests.get(f'{instance_url}/api/v1/instance')
            if response.status_code == 200:
                try:
                    instance_info = response.json()
                    #print(instance_info)
                    rules = instance_info.get('rules', 'Data not available')
                    instance_rules.append(rules)
                    rule_counts.append(len(rules))
                except ValueError:  # Catch JSON decode errors
                    instance_rules.append("Data not available")
                    rule_counts.append(0)
            else:
                print(f"Failed to retrieve instance information for {instance_url}. Status code:", response.status_code)
                instance_rules.append(f"Failed to retrieve instance information. Status code: {response.status_code}")
                rule_counts.append(0)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while accessing {instance_url}: {e}")
            instance_rules.append("Unknown error")
            rule_counts.append(0)

    instance_df['rules'] = instance_rules
    
    instance_rule_count_data = {
        "instance" : list(instance_df['Instance Name']), 
        "instance group": list(instance_df['instance_group']),
        "rule count": rule_counts
    }
    print(instance_rule_count_data)
    instance_rule_count_df = pd.DataFrame(instance_rule_count_data)
    instance_rule_count_df.to_csv('instance_count.csv')

    cleaned_rules = []
    for rule in instance_rules: 
        instance_ruling = []
        if isinstance(rule, list):  # Ensure rule is a list before iteration
            for text in rule: 
                if isinstance(text, dict):  # Ensure text is a dictionary
                    instance_ruling.append(text.get('text', 'Key "text" not found'))
                else:
                    instance_ruling.append("Data type is not a dictionary")
        elif isinstance(rule, dict):  # Handle the case where rule is a dictionary
            instance_ruling.append(rule.get('text', 'Key "text" not found'))
        else:
            instance_ruling.append("Rule is not a list or dictionary")
        cleaned_rules.append(instance_ruling)
        
    instance_df['cleaned_rules'] = cleaned_rules

    print(instance_df.head())
    output_filename = 'instance_rules.csv'
    instance_df.to_csv(output_filename, index=False)
    return instance_df, output_filename


if __name__ == '__main__': 
    df_1_5, df_csv_1_5 = collect_instances(1, 5)
    df_1_5['instance_group'] = '1 to 5'
    print(len(df_1_5))
    time.sleep(10)

    df_6_15, df_csv_6_15 = collect_instances(6, 15)
    df_6_15['instance_group'] = '6 to 15'
    print(len(df_6_15))
    time.sleep(10)

    df_16_50, df_csv_16_50 = collect_instances(16, 50)
    df_16_50['instance_group'] = '16 to 50'
    print(len(df_16_50))
    time.sleep(10)

    df_51_150, df_csv_51_150 = collect_instances(51, 150)
    df_51_150['instance_group'] = '51 to 150'
    print(len(df_51_150))
    time.sleep(10)

    df_151_500, df_csv_151_500 = collect_instances(151, 500)
    df_151_500['instance_group'] = '151 to 500'
    print(len(df_151_500))
    time.sleep(10)

    df_501_1500, df_csv_501_1500 = collect_instances(501, 1500)
    df_501_1500['instance_group'] = '501 to 1500'
    print(len(df_501_1500))
    time.sleep(10)

    df_1501_5000, df_csv_1501_5000 = collect_instances(1501, 5000)
    df_1501_5000['instance_group'] = '1501 to 5000'
    print(len(df_1501_5000))
    time.sleep(10)

    df_5001, df_csv_5001 = collect_instances_min(5001)
    df_5001['instance_group'] = '5001+'
    print(len(df_5001))
    time.sleep(10)

    # Create one dataframe for all instances
    full_dataframe = pd.concat([df_1_5, df_6_15, df_16_50, df_51_150, df_151_500, df_501_1500, df_1501_5000, df_5001])
    full_dataframe.reset_index(drop=True, inplace=True)

    # Print the first few rows and the length of the full dataframe
    print(full_dataframe.head())
    print(len(full_dataframe))
    full_dataframe.to_csv('Complete_dataset.csv')

    #Stratified sampling for instance selection 

    sampled_dataset, test_df = train_test_split(full_dataframe, test_size=0.2, stratify=full_dataframe['instance_group'], random_state=42)
    sampled_dataset.to_csv('final_dataset.csv')

    df = pd.read_csv('final_dataset.csv')
    

    #Get rules 
    rules, rules_csv = get_rules(df)
    

