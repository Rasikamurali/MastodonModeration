from googletrans import Translator, LANGUAGES
import pandas as pd 
import numpy as np 
import os 
from matplotlib import pyplot as plt 
import requests

# #data1 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\1_15_final_formatted_data.csv')
# data2= pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\16_35_final_formatted_data.csv')
# #data3 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\36_150_final_formatted_data.csv')
# #data4 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\151_500_final_formatted_data.csv')
# #data5 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\501_1500_final_formatted_data.csv')
# #data6 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\1501_5000_final_formatted_data.csv')
# #data7 =pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\5001_final_formatted_data.csv')

# # Store the DataFrames in a list
# dataframes = [data2]

data = pd.read_csv(r'Formatted_data0.csv')


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

# for i, df in enumerate(dataframes, start=1):
#     translated_texts = [] 
#     for _,row in df.iterrows(): 
#         if row['lang'] == 'non-English' or row['lang'] == 'non-English': 
#             translated_texts.append(translate_text(row['preprocessed rules']))
#         else: 
#             translated_texts.append(row['preprocessed rules'])
#     df['translated text'] = translated_texts
#     df.to_csv(f'{i}_translated_final.csv')

#sample = data1.sample(n=50)
translated_text = []
for _, row in data.iterrows(): 
    #print(row['preprocessed rules'])
    if row['lang'] == 'non-English': 
        translated_text.append(translate_text(row['preprocessed rules']))
    else: 
        translated_text.append(row['preprocessed rules'])
# print(translated_text)
# print(len(translated_text))
data['translated rules'] = translated_text
data.to_csv('complete_translated_data.csv')



    



