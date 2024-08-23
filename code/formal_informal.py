import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
from transformers import pipeline

# data1 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\1_15_final.csv')
# data2= pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\16_35_final.csv')
# data3 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\36_150_final.csv')
# data4 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\151_500_final.csv')
# data5 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\501_1500_final.csv')
# data6 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\1501_5000_final.csv')
# data7 =pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\translated data\5001_final.csv')

data = pd.read_csv('sampled_dataset.csv')
print(data.head())
formality_classifier = pipeline("text-classification", model="s-nlp/xlmr_formality_classifier")
genre_classifier = pipeline("text-classification", model="classla/xlm-roberta-base-multilingual-text-genre-classifier")


# Store the DataFrames in a list
# dataframes = [data1,data2, data3, data4, data5, data6, data7]
# names = ['1_15', '16_35', '36_150', '151_500', '501_1500', '1501_5000', '5001']

# Iterate over the list and print the head of each DataFrame

formality_label = [] 
formality_score = [] 
genre_label = []


texts = data['translated rules']
for text in texts: 
        #print(pipe(text))
        formality_label.append(formality_classifier(text)[0]['label'])
        formality_score.append(formality_classifier(text)[0]['score'])
        genre_label.append(genre_classifier(text)[0]['label'])
        
data['formality label'] = formality_label
data['formality score'] = formality_score
data['genre label'] = genre_label

print(data.head())
print(data.columns)

data.to_csv('formality_sampled_dataset.csv')
