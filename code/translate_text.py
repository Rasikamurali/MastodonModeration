from googletrans import Translator, LANGUAGES
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 


data = pd.read_csv(r'C:\Users\rasik\Documents\IS\1_15_final_formatted_data.csv')
print(data.head())
print(data.columns)

def translate_text(text, dest_lang='en'):
    translator = Translator()
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print(f"Error: {e}")
        return None

# langugaes = list(data['lang']) 

# for lang in langugaes: 
#     if lang == 'non-Englsih': 
#         print("not en")
translated_text = []
for _, row in data.iterrows(): 
    if row['lang'] != 'non-Englsih': 
        #print(row)
        translated_text.append(translate_text(row['preprocessed rules']))




# # Example usage
# text_to_translate = "Hola, ¿cómo estás?"
# translated_text = translate_text(text_to_translate)

# if translated_text:
#     print(f"Original text: {text_to_translate}")
#     print(f"Translated text: {translated_text}")
# else:
#     print("Translation failed.")

