import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


one = pd.read_csv("First_20_rules.csv")
two = pd.read_csv("Second_20_rules.csv")
three = pd.read_csv("Third_50_rules.csv")

appended_df = one._append(two)
df = appended_df._append(three)

#print(df.head())
print(df.columns)
new_df = df[['Instance Name', 'User Count', 'Short', 'rules']]
#print(new_df.head())

rules = (df['rules'])

cleaning_rules = []
for rule in rules: 
    #print(rule)
    if rule == '[]': 
        print("empty")
        cleaning_rules.append('0')
    elif rule == 'Rule not a field': 
        cleaning_rules.append('0')
    elif rule == 'Data not available': 
        cleaning_rules.append('0')
    else: 
        cleaning_rules.append(rule)

print(cleaning_rules)
new_df['new_rules'] = cleaning_rules
print(new_df.head())

filtered_instances = new_df[new_df['new_rules'] == '0']
print(filtered_instances[['Instance Name', 'User Count', 'Short']])

# Define bins for the user count ranges
bins = [0, 5000, 10000, float('inf')]
labels = ['Less than 5000', '5000-10000', '10000+']

# Add a new column with the user count ranges
filtered_instances['User Count Range'] = pd.cut(filtered_instances['User Count'], bins=bins, labels=labels, right=False)

counts = filtered_instances['User Count Range'].value_counts().sort_index()

plt.bar(counts.index, counts.values)
plt.xlabel('User Count Range')
plt.ylabel('Count')
plt.title('User count range Distribution')
plt.show()


# Tokenization
tokens = [word for rule in rules for word in rule.lower().split()]


# Frequency Analysis
token_counts = Counter(tokens)
print(token_counts)

# # Visualization
# df_token_counts = pd.DataFrame.from_dict(token_counts, orient='index', columns=['Count'])
# df_token_counts = df_token_counts.sort_values(by='Count', ascending=False)
# df_token_counts.plot(kind='bar')
# plt.xlabel('Token')
# plt.ylabel('Frequency')
# plt.title('Token Frequency Analysis')
# plt.show()