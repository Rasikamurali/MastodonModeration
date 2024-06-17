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

def full_cleaning(df): 
    df = pd.read_csv(df)
    df = df[df['rules'] != '[]']
    df = df[df['rules'] != 'Data not available']
    df = df[df['rules'] != 'Unknown error']
    df = df[df['rules'] != 'Failed to retrieve instance information. Status code: 502']
    df = df[df['rules'] != 'Rule is not a list']
    df['cleaned_rules'] = df['cleaned_rules'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    print(df.head())
    print(len(df))
    return df


def remove_punctuations(text):
    text = text.lower()
    # Define a regex pattern to match all punctuations
    pattern = r'[^\w\s]'
    # Substitute punctuations with an empty string
    clean_text = re.sub(pattern, '', text)

    clean_text = BeautifulSoup(clean_text, "html.parser").get_text()
    
    # Remove Markdown notations and other escape sequences like \n, \r, etc.
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple whitespace with single space
    clean_text = re.sub(r'\\n', ' ', clean_text)  # Replace new line notations
    clean_text = re.sub(r'\\r', ' ', clean_text)  # Replace carriage return notations

    # Strip leading and trailing whitespace
    clean_text = clean_text.strip()
    return clean_text

# Define a function to transform the data
def transform_data(df):
    # Initialize an empty list to store the transformed data
    transformed_data = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Parse the rules from string to list if they are in string format
        rules = eval(row['cleaned_rules']) if isinstance(row['cleaned_rules'], str) else row['cleaned_rules']
        
        # Create a new row for each rule
        for rule in rules:
            transformed_data.append({'instance': row['Instance Name'], 'rule': rule})

    # Create a new DataFrame from the transformed data
    new_df = pd.DataFrame(transformed_data)
    
    return new_df


df = full_cleaning(r'C:\Users\rasik\Documents\IS\instance_rules_5001.csv')
# Transform the dataset
new_df = transform_data(df)
# print(df.columns)
print(new_df.head())

rules_rules = new_df['rule']
preprocessed_rules= [] 

for rules in rules_rules: 
     preprocessed_rules.append(remove_punctuations(rules))

lang_detect = []
for rule in preprocessed_rules: 
    try:
        language = detect(rule)

        if language != 'en': 
            lang_detect.append("non-Englsih")
        else: 
            lang_detect.append("English")
    except Exception as e:
        lang_detect.append("Error")


print(Counter(lang_detect))
print(len(new_df), len(preprocessed_rules))
new_df['preprocessed rules'] = preprocessed_rules
new_df['lang'] = lang_detect

final_df = new_df[new_df['lang'] != 'Error']

final_df = final_df[final_df['preprocessed rules'] != 'Rule is not a list']
print(final_df.head())
print(len(final_df))

final_df.to_csv('5001_final_formatted_data.csv')





