# weekly_activity_data_collection.py
#
# Fetches weekly activity data (statuses, logins, registrations) from each
# Mastodon instance's public activity API endpoint (/api/v1/instance/activity).
# Data is summed per instance and saved for use as an engagement proxy in
# robustness analyses.
#
# Input:  complete_strat20_translated_rules.csv  (stratified sample of instances)
# Output: strat20_weeklyactivity_data.csv

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
from datetime import datetime, timedelta


# Load the stratified sample and extract unique instance names
df = pd.read_csv(r'complete_strat20_translated_rules.csv')
print(df.columns)
df = df.rename(columns={'instance': 'Instance Name'})
columns = ['Instance Name']
df1 = df[columns]

df1 = df1.drop_duplicates(subset='Instance Name')

df_sample = df1.copy()

df_sample = df_sample[df_sample['Instance Name'] != 'stream.nik.mx']
df_sample = df_sample[df_sample['Instance Name'] != 'kooktzi.ch']
print(df_sample)


# Initialize lists to store results
instance_names = []
weeks = []
statuses_sum = []
logins_sum = []
registrations_sum = []

# Iterate through the instances in df_sample
for index, row in df_sample.iterrows():
    instance_name = row['Instance Name']
    instance_url = f'https://{instance_name}/api/v1/instance/activity'

    try:
        # Fetch data from the API
        response = requests.get(instance_url)
        if response.status_code == 200:
            try:
                # Parse JSON response
                instance_info = response.json()
                
                # Check if data is complete and valid
                if isinstance(instance_info, list) and all(
                    isinstance(entry, dict) and
                    'statuses' in entry and
                    'logins' in entry and
                    'registrations' in entry
                    for entry in instance_info
                ):
                    for entry in instance_info:
                        instance_names.append(instance_name)
                        statuses_sum.append(int(entry.get('statuses', 0)))
                        logins_sum.append(int(entry.get('logins', 0)))
                        registrations_sum.append(int(entry.get('registrations', 0)))
                else:
                    print(f"Incomplete data for {instance_name}. Skipping...")
            except Exception as json_error:
                print(f"JSON parsing error for {instance_name}: {json_error}")
        else:
            print(f"Non-200 response for {instance_name}, status code: {response.status_code}")
    except requests.exceptions.RequestException as req_error:
        print(f"Request error for {instance_name}: {req_error}")



# Initialize result_df to None
result_df = None

# Ensure all lists have the same length before creating the DataFrame
if len(instance_names) == len(statuses_sum) == len(logins_sum) == len(registrations_sum):
    # Create DataFrame
    result_df = pd.DataFrame({
        'Instance Name': instance_names,
        'Statuses': statuses_sum,
        'Logins': logins_sum,
        'Registrations': registrations_sum
    })
    print(result_df)
else:
    print("Mismatched data lengths. Skipping DataFrame creation.")

# Check if result_df is defined and not empty
if result_df is not None and not result_df.empty:
    # Group data by 'Instance Name'
    final_df = result_df.groupby(['Instance Name'], as_index=False).sum()

    # Select and rename columns for clarity
    final_df = final_df.rename(columns={
        'Instance Name': 'Instance',
        'Statuses': 'Total Statuses',
        'Logins': 'Total Logins',
        'Registrations': 'Total Registrations'
    })

    # Print columns and the final DataFrame for debugging
    print(final_df.columns)
    print(final_df)
    final_df.to_csv('strat20_weeklyactivity_data.csv')
else:
    print("Skipping final DataFrame creation due to missing or invalid data.")
