import requests
import json
import pandas as pd

# URL of the Mastodon instance you want to get information from

instances = ['genomic.social', 'ruhr.social', 'mastodontech.de', 'sfba.social', 'kolektiva.social', 'mastodon.cloud', 'mastodon.green', 'mastodon.social']

instance_description = [] 
instance_rules = [] 
instance_user_count = [] 
instance_short_desc = [] 
for instance in instances: 

    instance_url = f'https://{instance}'  # Replace with the URL of the instance you're interested in
    print(instance_url)
    # Make a GET request to the instance API
    response = requests.get(f'{instance_url}/api/v1/instance')

# Check if the request was successful
    if response.status_code == 200:
        try:
            instance_info = response.json()
            #print(instance_info)
        # Print instance description
            #print("Instance description:", instance_info['rules'])
            instance_short_desc.append(instance_info['short_description'])
            instance_user_count.append(list(instance_info['stats'].values())[0])
            instance_description.append(instance_info['description'])
            instance_rules.append(instance_info['rules'])
        except requests.exceptions.JSONDecodeError: 
            print("JSON not in expected format")
            instance_description.append("Data not available")
            instance_rules.append("Data not available")
            instance_short_desc.append("Data not available")
            instance_user_count.append("Data not available")
    else:
        print("Failed to retrieve instance information. Status code:", response.status_code)

print(len(instances))
print(len(instance_description))
print(len(instance_rules))
print(len(instance_user_count))
print(len(instance_short_desc))


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

data = {'instance' : instances, 
        'desc' : instance_description, 
        'short desc': instance_short_desc,
        'rules' : cleaned_rules, 
        'user count': instance_user_count}
#about_and_rules_df = pd.DataFrame(instance_rules, instance_description, columns =['about', 'rules'])
#print(about_and_rules_df.head())
df = pd.DataFrame(data)
print(df.head())
df.to_csv('instance_details.csv')

