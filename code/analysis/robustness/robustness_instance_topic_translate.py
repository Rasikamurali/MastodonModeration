# robustness_instance_topic_translate.py
#
# Translates instance short descriptions from non-English to English using
# langdetect for language identification and googletrans for translation.
# The translated descriptions are used for GPT-based instance topic categorization.
#
# Input:  data/mstdn_topics_wnorms.csv        (instance metadata with short descriptions)
# Output: data/instance_topics_translated.csv  (short descriptions with translated text)

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
from googletrans import Translator, LANGUAGES

# Load instance metadata containing short descriptions to be translated
mstdn_topics_wnorms = pd.read_csv(r'data/mstdn_topics_wnorms.csv')
print(mstdn_topics_wnorms.columns)

instance_topics = mstdn_topics_wnorms[['Instance Name', 'Description', 'Short', 'topic']].drop_duplicates()

topics = instance_topics['Short'].tolist()

lang_detect = []
for topic in topics: 
    try:
        language = detect(topic)

        if language != 'en': 
            lang_detect.append("non-English")
        else: 
            lang_detect.append("English")
    except Exception as e:
        lang_detect.append("Error")

# #Then we translate everything 

instance_topics['lang'] = lang_detect

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
    
translated_texts=[]
for _,row in instance_topics.iterrows(): 
        if row['lang'] == 'non-English' or row['lang'] == 'non-English': 
            translated_texts.append(translate_text(row['Short']))
        else: 
            translated_texts.append(row['Short'])
instance_topics['translated short'] = translated_texts

instance_topics.to_csv(r'data/instance_topics_translated.csv', index=False)