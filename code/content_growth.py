import pandas as pd 
import numpy as np 
import os
import nltk
nltk.download('stopwords')
from collections import Counter
import ast 
import regex as re 
nltk.download('punkt')
from bs4 import BeautifulSoup
from langdetect import detect
from matplotlib import pyplot as plt 
import json
import gzip

def count_updates_in_gz_files(directory):
    """
    Counts the number of "event_type" == "update" in each json.gz file within the specified directory.
    
    Args:
    - directory (str): The path to the directory containing the json.gz files.
    
    Returns:
    - dict: A dictionary where the keys are the names of the json.gz files and the values are the counts of "update" event types.
    """
    # Initialize a dictionary to store the count for each json.gz file
    gz_update_counts = {}

    # Loop through all files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a json.gz file
        if item.endswith('.json.gz'):
            # Initialize the count for this json.gz file
            update_count = 0
            
            # Open the gzipped JSON file
            with gzip.open(item_path, 'rt', encoding='utf-8') as gz_file:
                buffer = ""
                for line in gz_file:
                    # Since the JSON objects are not separated by newlines, append each line to a buffer
                    buffer += line.strip()
                    
                    try:
                        # Try to decode the JSON from the buffer
                        data = json.loads(buffer)
                        
                        # Check if "event_type" is "update" and count it
                        if "event_type" in data and data["event_type"] == "update":
                            update_count += 1
                        
                        # Clear the buffer after successfully reading one JSON object
                        buffer = ""
                    
                    except json.JSONDecodeError:
                        # If JSONDecodeError occurs, continue to the next line
                        # This assumes that the error is due to incomplete JSON data in the buffer
                        continue
            
            # Store the count in the dictionary with the json.gz file name as the key
            gz_update_counts[item] = update_count

    return gz_update_counts

directory = r"C:\Users\rasik\Downloads\Mastodon_data\Mastodon_data\2024-08-27"
result1 = count_updates_in_gz_files(directory)
directory1 = r"C:\Users\rasik\Downloads\Mastodon_data\Mastodon_data\2024-08-28"
result2 = count_updates_in_gz_files(directory1)

def add_dict_values(dict1, dict2):
    """
    Adds values from two dictionaries with the same keys.
    
    Args:
    - dict1 (dict): The first dictionary.
    - dict2 (dict): The second dictionary.
    
    Returns:
    - dict: A dictionary with the same keys, where values are the sum of values from dict1 and dict2.
    """
    # Initialize a new dictionary to store the result
    result = {}
    
    # Loop through the keys in the first dictionary
    for key in dict1:
        if key in dict2:
            result[key] = dict1[key] + dict2[key]
        else:
            # Handle case where key is in dict1 but not in dict2
            result[key] = dict1[key]
    
    # Optionally, handle case where dict2 has keys not in dict1
    for key in dict2:
        if key not in dict1:
            result[key] = dict2[key]
    
    return result


result = add_dict_values(result1, result2)
print(result)

instances = list(result.keys())


def clean_filenames_in_dict(input_dict):
    """
    Cleans the filenames (keys) in the dictionary by removing the date and '.json.gz' extension.
    
    Args:
    - input_dict (dict): The original dictionary with filenames as keys.
    
    Returns:
    - dict: A new dictionary with cleaned keys.
    """
    cleaned_dict = {}
    
    for filename, value in input_dict.items():
        # Clean the filename
        cleaned_name = re.sub(r'_\d{4}-\d{2}-\d{2}\.json\.gz$', '', filename)
        # Add the cleaned filename and its value to the new dictionary
        cleaned_dict[cleaned_name] = value
    
    return cleaned_dict

cleaned_dict = clean_filenames_in_dict(result)

final_dataset = pd.read_csv(r'final_dataset.csv')

mastodon_instances = [
    "mastodon.social",
    "pawoo.net",
    "mstdn.jp",
    "mstdn.social",
    "infosec.exchange",
    "mastodon.online",
    "mas.to",
    "hachyderm.io",
    "mastodon.world",
    "troet.cafe",
    "mastodon.gamedev.place",
    "piaille.fr",
    "aethy.com",
    "planet.moe",
    "techhub.social",
    "mastodon.uno",
    "mastodon.art",
    "social.vivaldi.net",
    "social.tchncs.de",
    "mastodon.nl",
    "mastodonapp.uk",
    "kolektiva.social",
    "mastodon.sdf.org",
    "c.im",
    "nrw.social",
    "sfba.social",
    "aus.social",
    "mastodon.cloud",
    "mastodon.scot",
    "occm.cc",
    "toot.community",
    "mstdn.party",
    "ohai.social",
    "ruhr.social",
    "ioc.exchange",
    "mastodon.nu",
    "kind.social",
    "mindly.social",
    "indieweb.social",
    "mastodon.green",
    "social.linux.pizza",
    "defcon.social",
    "nerdculture.de",
    "mastodon.au",
    "livellosegreto.it",
    "g0v.social",
    "mastodont.cat",
    "zirk.us",
    "sigmoid.social",
    "ecoevo.social",
    "mastouille.fr",
    "metalhead.club",
    "sciences.social",
    "tkz.one",
    "genomic.social",
    "mstdn.business",
    "stranger.social",
    "mast.lat",
    "mastodon.org.uk"
]


for instance in mastodon_instances:
    # Check if the instance exists in the 'Instance Name' column of final_dataset
    matching_rows = final_dataset[final_dataset['Instance Name'] == instance]
    
    # If there's a match, print the corresponding 'User Count'
    if not matching_rows.empty:
        print(matching_rows['User Count'].values[0])
# Step 1: Convert the dictionary to a DataFrame
dict_df = pd.DataFrame(list(cleaned_dict.items()), columns=['Instance Name', 'value'])

# Step 2: Merge the DataFrame with the dictionary DataFrame on 'instance_name'
merged_df = pd.merge(final_dataset, dict_df, on='Instance Name', how='inner')

# dict_df.to_csv('tester101.csv')
# merged_df.to_csv('tester102.csv')

# # Step 3: Now, merged_df contains 'instance_name', 'num_users', and 'value'
# # You can plot value vs num_users
import matplotlib.pyplot as plt

plt.scatter(merged_df['User Count'], merged_df['value'])
plt.xlabel('Number of Users')
plt.ylabel('Value')
plt.yscale('log')
plt.xscale('log')
plt.title('Value vs Number of Users')
plt.show()
