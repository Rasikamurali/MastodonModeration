import pandas as pd 
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

rules_df = pd.read_csv('fifteen_thirtyfive_user_rules.csv')
rules = rules_df['cleaned_rules']

cleaned_rules = []
for rule in rules: 
    if rule != '[]':
        cleaned_rules.append(rule)
    

for rule in cleaned_rules: 
    rule = rule.lower() 
    print(rule)

text = ' '.join(cleaned_rules)
text= text.replace('string', '')
text= text.replace('type', '')
text= text.replace('data', '')

# Generate word cloud
wordcloud = WordCloud().generate(text)

# Display the word cloud using matplotlib
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

        
    