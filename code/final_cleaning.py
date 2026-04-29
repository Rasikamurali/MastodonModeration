import pandas as pd 
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from collections import Counter
import ast 
import regex as re 
nltk.download('punkt')
from bs4 import BeautifulSoup
from langdetect import detect

df = pd.read_csv(r'C:\Users\rasik\Documents\IS\instance_rules_1_15.csv')
print(df.head())
print(df.columns)
print(Counter(df['rules']))
print(len(df))

df = df[df['rules'] != '[]']
print(df.head())
print(len(df))

df = df[df['rules'] != 'Data not available']
print(len(df))

df = df[df['rules'] != 'Unknown error']
print(len(df))

df = df[df['rules'] != 'Failed to retrieve instance information. Status code: 502']
print(len(df))


print(len(df['cleaned_rules']))
#print(Counter(df['cleaned_rules']))

df = df[df['rules'] != 'Rule is not a list']

print(len(df))

cleaned_rules = list(df['cleaned_rules']) 
converted_rules_list = [ast.literal_eval(rule) for rule in cleaned_rules]

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


preprocessed_rules= [] 
json_rules = [] 
formatted_rules = [] 
print(len(df['rules']))
print(len(df['cleaned_rules']))


for rules in converted_rules_list: 
    #print(rules)
    for rule in rules: 
        if rule != 'Rule is not a list': 
            formatted_rules.append(rule)
            preprocessed_rules.append(remove_punctuations(rule))


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
print(len(preprocessed_rules))
data = { 
    'formatted_rules': formatted_rules, 
    'cleaned_rules': preprocessed_rules, 
    'lang': lang_detect
}
new_df = pd.DataFrame(data)
print(new_df.head())

new_df.to_csv('Tester.csv')

test_df = new_df[new_df['lang'] == 'Error']
print(test_df.head())
test_df.to_csv('checker.csv')

final_df = new_df[new_df['lang'] != 'Error']

final_df.to_csv('5001_final_formatted_data.csv')






#new_df.to_csv('50001_formatted_rules.csv')