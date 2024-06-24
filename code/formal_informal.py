import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
from transformers import pipeline


data1 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\1_15_final_formatted_data.csv')
data2= pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\16_35_final_formatted_data.csv')
data3 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\36_150_final_formatted_data.csv')
data4 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\151_500_final_formatted_data.csv')
data5 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\501_1500_final_formatted_data.csv')
data6 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\1501_5000_final_formatted_data.csv')
data7 =pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\5001_final_formatted_data.csv')


pipe = pipeline("text-classification", model="s-nlp/xlmr_formality_classifier")
pipe2 = pipeline("text-classification", model="classla/xlm-roberta-base-multilingual-text-genre-classifier")


# Store the DataFrames in a list
dataframes = [data1,data2, data3, data4, data5, data6, data7]

# Iterate over the list and print the head of each DataFrame
for i, df in enumerate(dataframes, start=1):
    formality_label = [] 
    formality_score = [] 
    genre_label = []
    genre_score = [] 
    print(df.head())
    texts = df['preprocessed rules']
    for text in texts: 
        #print(pipe(text))
        formality_label.append(pipe(text)[0]['label'])
        formality_score.append(pipe(text)[0]['score'])
        genre_label.append(pipe2(text)[0]['label'])
        genre_score.append(pipe2(text)[0]['score'])
    df['formality label'] = formality_label
    df['formality score'] = formality_score
    df['genre label'] = genre_label
    df['genre score'] = genre_label
    df.to_csv(f'{i}_formality_genre.csv')
