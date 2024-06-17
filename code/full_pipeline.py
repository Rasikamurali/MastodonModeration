import pandas as pd 
import numpy as np 
import os 
import requests 
from matplotlib import pyplot as plt 
import requests 
import urllib3
import nltk 
import json


#Id and token for Instances API to collect instances 
id = '796768437'
id_token = 'ZMaefRB4oUfPcOx892VsAlWuU5sIYlGWXEiw7DVYt0AXDVUsPBSjsIegFol0EBIm0hKz2NWv7I1WKofhysQlWMoXGWHcJ6z6GJlKhcKXD5FqQ537Bvz0A7Bq3wvAOwHN'


#Number of instances I want to retrieve 
count = 10000 

#Defining function to collect instances 

def collect_instances(min_users, include_down=False): 
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
                active_users = instance.get('info').get('active_users')
                reg = instance.get('info').get('dead') 
                closed = instance.get('open_registrations')

                instance_data.append({
                        "Instance Id": id,
                        "Instance Name": name,
                        "Is Dead": dead,
                        "User Count": user_count,
                        "Description": description,
                        "Short": short_description,
                        "topic": topic,
                        "active users": active_users,
                        "dead": reg, 
                        "reg": closed
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

    for index, row in instance_df.iterrows():
        instance_name = row['Instance Name']
        print(instance_name)
        instance_url = f'https://{instance_name}'

        try:
            response = requests.get(f'{instance_url}/api/v1/instance')
            if response.status_code == 200:
                try:
                    instance_info = response.json()
                    instance_rules.append(instance_info.get('rules', 'Data not available'))
                except ValueError:  # Catch JSON decode errors
                    instance_rules.append("Data not available")
            else:
                print(f"Failed to retrieve instance information for {instance_url}. Status code:", response.status_code)
                instance_rules.append(f"Failed to retrieve instance information. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while accessing {instance_url}: {e}")
            instance_rules.append("Unknown error")

    instance_df['rules'] = instance_rules

    cleaned_rules = []
    for rule in instance_rules: 
        instance_ruling = []
        if isinstance(rule, list):  # Ensure rule is a list before iteration
            for text in rule: 
                if isinstance(text, dict):  # Ensure text is a dictionary
                    instance_ruling.append(list(text.values())[-1])
                else:
                    instance_ruling.append("Data type is not a dictionary")
        else:
            instance_ruling.append("Rule is not a list")
        cleaned_rules.append(instance_ruling)
        
    instance_df['cleaned_rules'] = cleaned_rules

    print(instance_df.head())
    output_filename = 'instance_rules.csv'
    instance_df.to_csv(output_filename, index=False)
    return instance_df, output_filename


if __name__ == '__main__': 
    # df, df_csv = collect_instances(1, 15)
    # print(df.head())
    # sampled_df = df.sample(n=1000)
    # rules, rules_csv = get_rules(sampled_df)
    # print(rules.head())

    # df1, df1_csv = collect_instances(16, 35)
    # print(df1.head())
    # rules1, rules1_csv = get_rules(df1)
    # print(rules1.head())

    # df2, df2_csv = collect_instances(36, 150)
    # print(df2.head())
    # rules2, rules2_csv = get_rules(df2)
    # print(rules2.head())

    # df3, df3_csv = collect_instances(151, 500)
    # print(df3.head())
    # rules3, rules3_csv = get_rules(df3)
    # print(rules3.head())

    # df4, df4_csv = collect_instances(501, 1500)
    # print(df4.head())
    # rules4, rules4_csv = get_rules(df4)
    # print(rules4.head())

    # df5, df5_csv = collect_instances(1501, 5000)
    # print(df5.head())
    # rules5, rules5_csv = get_rules(df5)
    # print(rules5.head())

    df6, df6_csv = collect_instances(5001)
    print(df6.head())
    rules6, rules6_csv = get_rules(df6)
    print(rules6.head())




    



# #Setting min and max users and excluding instances that are down, limiting language to english 
# min_users= 1500
# max_users = 5000
# include_down = False 
# language = 'English'
# #URL for API 
# url = f"https://instances.social/api/1.0/instances/list?count={count}&min_users={min_users}&max_users={max_users}&language={language}"

# # Create headers with the Authorization header
# headers = {
#     'Authorization': f'Bearer {id_token}'
# }

# try:
#     response = requests.get(url, headers=headers)

#     if response.status_code == 200:
#         response_data = response.content.decode('utf-8')  # Decode the bytes response to a string
#         data = json.loads(response_data)  # Parse the JSON data as a list
#         #print(data)
#         #print(data)
#         # Extract relevant information from the response and create a list of dictionaries
#         instance_data = []
#         instances = data.get('instances')
#         for instance in instances:
#           id = instance.get('id')
#           name = instance.get('name')
#           dead = instance.get('dead')
#           user_count = instance.get('users')
#           description = instance.get('info').get('full_description')
#           short_description = instance.get('info').get('short_description')
#           topic =  instance.get('info').get('topic')
#           active_users = instance.get('info').get('active_users')
#           reg = instance.get('info').get('dead') 
#           closed = instance.get('open_registrations')

#           instance_data.append({
#                 "Instance Id": id,
#                 "Instance Name": name,
#                 "Is Dead": dead,
#                 "User Count": user_count,
#                 "Description": description,
#                 "Short": short_description,
#                 "topic": topic,
#                 "active users": active_users,
#                 "dead": reg, 
#                 "reg": closed
#             })

#         # Create a DataFrame from the list of dictionaries
#         df = pd.DataFrame(instance_data)
#         print(df)


#         # Print the DataFrame
#         print(df)
#     else:
#         print(f"Error: {response.status_code}")
# except Exception as e:
#     print(f"Error: {e}")

# df.to_csv('Instances_maxusers_fifteenhundred_fivethousand.csv')
# print(df.head())
# print(len(df))
    

# #Once we get the list of instances, we want to collect the rules for those instances 

# # Read the DataFrame
# df = pd.read_csv('Instances_maxusers_five_fifteenhundred.csv')

# instances = df['Instance Name']
# instance_rules = []

# #random sampling of 100 instances 
# sampled_instances = df.sample(n=40)
# print(sampled_instances.head())
# #Getting all policies
# count = 0 

# for index, row in sampled_instances.iterrows():
#     print(row['Instance Name']) 
#     instance_url = f'https://{row['Instance Name']}'  
#     try:
#         response = requests.get(f'{instance_url}/api/v1/instance')

#         if response.status_code == 200:
#             try:
#                 instance_info = response.json()
#                 instance_rules.append(instance_info.get('rules', 'Data not available'))
#             except requests.exceptions.JSONDecodeError: 
#                 instance_rules.append("Data not available")
#             except KeyError: 
#                 instance_rules.append("Rule not a field")
#         else:
#             print(f"Failed to retrieve instance information for {instance_url}. Status code:", response.status_code)
#             instance_rules.append(f"Failed to retrieve instance information. Status code: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred while accessing {instance_url}: {e}")
#         instance_rules.append("Unknown error")

# # Continue with the rest of your code...
# print(instance_rules)

# sampled_instances['rules'] = instance_rules

# cleaned_rules = []
# for rule in instance_rules: 
#     instance_ruling = []
#     for text in rule: 
#         try: 
#             instance_ruling.append((list(text.values())[-1]))
#         except: 
#             instance_ruling.append(("data type is a string"))
#     cleaned_rules.append(instance_ruling)
# print(cleaned_rules)

# sampled_instances['cleaned_rules'] = cleaned_rules

# print(sampled_instances.head())
# sampled_instances.to_csv('five_fifteenhundred_user_rules.csv')