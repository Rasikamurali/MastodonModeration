import numpy as np 
import pandas as pd 
import os 
from matplotlib import pyplot as plt 
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string 
from lexicalrichness import LexicalRichness
import textstat

#splitting the dfs 
data1 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\1_15_final_formatted_data.csv')
data2= pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\16_35_final_formatted_data.csv')
data3 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\36_150_final_formatted_data.csv')
data4 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\151_500_final_formatted_data.csv')
data5 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\501_1500_final_formatted_data.csv')
data6 = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\1501_5000_final_formatted_data.csv')
data7 =pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\5001_final_formatted_data.csv')

pre_150 = pd.concat([data1, data2, data3])
#print(len(pre_150))

post_150 = pd.concat([data4, data5, data6, data7])
#print(len(post_150))

#Word Count 
def word_count(pre_150, post_150):
    pre_150_wc = [] 
    post_150_wc = []
    for _, row in pre_150.iterrows(): 
        pre_150_wc.append(len(row['preprocessed rules']))
    for _, row in post_150.iterrows(): 
        post_150_wc.append(len(row['preprocessed rules']))
    return pre_150_wc, post_150_wc

pre_150_wc_list, post_150_wc_list = word_count(pre_150, post_150)
t_statistic, p_value = stats.ttest_ind(pre_150_wc_list, post_150_wc_list)
print("word count")
print(t_statistic, p_value)

#Type Token Ratio

def ttr(pre_150, post_150): 
    pre_150_ttr = [] 
    post_150_ttr = [] 
    for _, row in pre_150.iterrows(): 
        tokens = [word for word in row['preprocessed rules'] if word.isalnum()]
        total_tokens = len(tokens)
        unique_tokens = set(tokens)
        total_types = len(unique_tokens)
        ttr = total_types / total_tokens
        pre_150_ttr.append(ttr)
    for _, row in post_150.iterrows(): 
        tokens = [word for word in row['preprocessed rules'] if word.isalnum()]
        total_tokens = len(tokens)
        unique_tokens = set(tokens)
        total_types = len(unique_tokens)
        ttr = total_types / total_tokens
        post_150_ttr.append(ttr)
    return pre_150_ttr, post_150_ttr

pre_150_ttr_list, post_150_ttr_list = ttr(pre_150, post_150)
t_statistic_ttr, p_value_ttr = stats.ttest_ind(pre_150_ttr_list, post_150_ttr_list)
print("ttr")
print(t_statistic_ttr, p_value_ttr)


#Lexical Density 

def lex_readibility(pre_150, post_150): 
    pre_150_lr = [] 
    post_150_lr = []
    for _, row in pre_150.iterrows(): 
        lex = LexicalRichness(row['preprocessed rules'])
        pre_150_lr.append(lex.Summer)
    for _, row in post_150.iterrows(): 
        lex = LexicalRichness(row['preprocessed rules'])
        post_150_lr.append(lex.Summer)
    return pre_150_lr, post_150_lr

pre_150_lr_list, post_150_lr_list = ttr(pre_150, post_150)
t_statistic_lr, p_value_lr = stats.ttest_ind(pre_150_lr_list, post_150_lr_list)
print("lr")
print(t_statistic_lr, p_value_lr)


#Readibility 
def word_count(pre_150, post_150):
    pre_150_rr = [] 
    post_150_rr = []
    for _, row in pre_150.iterrows(): 
        pre_150_rr.append(textstat.flesch_kincaid_grade(row['preprocessed rules']))
    for _, row in post_150.iterrows(): 
        post_150_rr.append(textstat.flesch_kincaid_grade(row['preprocessed rules']))
    return pre_150_rr, post_150_rr

pre_150_rr_list, post_150_rr_list = word_count(pre_150, post_150)
t_statistic_rr, p_value_rr = stats.ttest_ind(pre_150_rr_list, post_150_rr_list)
print("Readibility")
print(t_statistic_rr, p_value_rr)