from googletrans import Translator, LANGUAGES
import pandas as pd 
import numpy as np 
import os 
from matplotlib import pyplot as plt 
import requests
import ast 
from langdetect import detect


#First we need to remove everything that is empty or that says 'Data not available' 
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
            transformed_data.append({'instance': row['Instance Name'], 'rule': rule, 'instance group': row['instance_group']})

    # Create a new DataFrame from the transformed data
    new_df = pd.DataFrame(transformed_data)
    
    return new_df

cleaned_df = full_cleaning('final_rules.csv')
print(len(cleaned_df))
transformed_df = transform_data(cleaned_df)
print(len(transformed_df))
print(transformed_df.columns)

#Then we identify non-English ones 
preprocessed_rules = transformed_df['rule']

lang_detect = []
for rule in preprocessed_rules: 
    try:
        language = detect(rule)

        if language != 'en': 
            lang_detect.append("non-English")
        else: 
            lang_detect.append("English")
    except Exception as e:
        lang_detect.append("Error")

# #Then we translate everything 

transformed_df['lang'] = lang_detect

# Initialize the translator
translator = Translator()

def translate_text(text, dest_lang='en'):
    # Initialize the Translator object
    translator = Translator()
    
    try:
        # Translate the text
        translated = translator.translate(text, dest=dest_lang)
        
        # Return the translated text
        #print(translated.text)
        return translated.text
    except Exception as e:
        # Print the error message if any exception occurs
        print(f"Error: {e}")
        return None

sample_df = transformed_df[16000:]

translated_texts=[]
for _,row in sample_df.iterrows(): 
        if row['lang'] == 'non-English' or row['lang'] == 'non-English': 
            translated_texts.append(translate_text(row['rule']))
        else: 
            translated_texts.append(row['rule'])
sample_df['translated text'] = translated_texts
sample_df.to_csv(f'translated_final2.csv')






    



