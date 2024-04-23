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
min_users= 1500
max_users = 5000
include_down = False 
url = f"https://instances.social/api/1.0/instances/list?count={count}&min_users={min_users}&max_users={max_users}"

# Create headers with the Authorization header
headers = {
    'Authorization': f'Bearer {id_token}'
}

try:
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
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
        print(df)


        # Print the DataFrame
        print(df)
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")

df.to_csv('Instances_maxusers_fifteenhundred_fivethousand.csv')
print(df.head())
print(len(df))
    
