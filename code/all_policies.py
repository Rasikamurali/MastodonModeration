import pandas as pd 
import numpy as np 
import os 
import requests 


id = '796768437'
id_token = 'ZMaefRB4oUfPcOx892VsAlWuU5sIYlGWXEiw7DVYt0AXDVUsPBSjsIegFol0EBIm0hKz2NWv7I1WKofhysQlWMoXGWHcJ6z6GJlKhcKXD5FqQ537Bvz0A7Bq3wvAOwHN'

import requests
import pandas as pd
import json

# Replace 'YOUR_SECRET_TOKEN' with your actual secret token
secret_token = id_token
count = 10000 # Change this to the number of instances you want to retrieve
#url = f"https://instances.social/api/1.0/instances/list?count={count}"
min_users = 1000
url = f"https://instances.social/api/1.0/instances/list?count={count}&min_users={min_users}"

# Create headers with the Authorization header
headers = {
    'Authorization': f'Bearer {id_token}'
}

# try:
#     response = requests.get(url, headers=headers)

#     if response.status_code == 200:
#         response_data = response.content.decode('utf-8')  # Decode the bytes response to a string
#         data = json.loads(response_data)  # Parse the JSON data as a list
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

#           instance_data.append({
#                 "Instance Id": id,
#                 "Instance Name": name,
#                 "Is Dead": dead,
#                 "User Count": user_count,
#                 "Description": description,
#                 "Short": short_description,
#                 "topic": topic,
#                 "active users": active_users
#             })

#         # Create a DataFrame from the list of dictionaries
#         df = pd.DataFrame(instance_data)


#         # Print the DataFrame
#         print(df)
#     else:
#         print(f"Error: {response.status_code}")
# except Exception as e:
#     print(f"Error: {e}")

# df.to_csv('Instances_minusers.csv')
    
df = pd.read_csv('Instances_minusers.csv')
print(len(df))
print(df.head())
print(df.columns)
print(df['User Count'])
instances = df['Instance Name']

instance_rules = []
#Getting all policies
for instance in instances[150:180]: 

    instance_url = f'https://{instance}'  # Replace with the URL of the instance you're interested in
    #print(instance_url)
    # Make a GET request to the instance API
    response = requests.get(f'{instance_url}/api/v1/instance')

# Check if the request was successful
    if response.status_code == 200:
        try:
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
            instance_rules.append("Rule not a field")
        except requests.exceptions.ConnectTimeout: 
            instance_rules.append("Timeout")
    else:
        print("Failed to retrieve instance information. Status code:", response.status_code)

print(instance_rules)
new_df = df[150:180]
new_df['rules'] = instance_rules
print(new_df.head())
new_df.to_csv('Fourth_30_rules.csv')