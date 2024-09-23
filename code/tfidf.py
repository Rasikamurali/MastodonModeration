import pandas as pd 
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import html
import regex as re
import ast
from scipy import stats
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string 
from lexicalrichness import LexicalRichness
import textstat
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


data = pd.read_csv(r'translated_dataset_final.csv')

data = data.dropna(subset=['translated text'])

def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        
        # Lowercase the text
        text = text.lower()
        
        # Remove HTML entities
        text = html.unescape(text)
        
        # Remove punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]

        # Join the cleaned tokens back into a single string
        cleaned_text = ' '.join(tokens)

        # Append the cleaned text to the list
        cleaned_list.append(cleaned_text)
    
    return cleaned_list

data = data.dropna(subset=['translated text'])

rules = data['translated text']

data['cleaned translated text'] = clean_text(rules)

print(data.head())

# # Group by 'instance_group'
# groups = data.groupby('instance group')

# # Initialize a dictionary to store the normalized TF-IDF for each group
# tfidf_per_group = {}

# # Loop over each group and compute the TF-IDF
# for group_name, group_data in groups:
#     # Get the 'rules' column for the current group
#     rules_text = group_data['cleaned translated text'].dropna().values
    
#     # Initialize TF-IDF vectorizer
#     tfidf = TfidfVectorizer()
    
#     # Fit and transform the 'rules' column into a TF-IDF matrix
#     tfidf_matrix = tfidf.fit_transform(rules_text)
    
#     # Convert the matrix to a DataFrame
#     tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    
#     # Normalize by the number of documents in the group (document frequency normalization)
#     num_docs = len(rules_text)
#     print(num_docs)
#     normalized_tfidf_df = tfidf_df / num_docs
    
#     # Sum TF-IDF scores and get top 25 words
#     top_tfidf_words = normalized_tfidf_df.sum().sort_values(ascending=False).head(25)
    
#     # Store the normalized TF-IDF for this group
#     tfidf_per_group[group_name] = {
#         'tfidf_matrix': normalized_tfidf_df,
#         'tfidf words': top_tfidf_words
#     }

# # Now you can access the normalized TF-IDF for each instance group
# for group_name, tfidf_data in tfidf_per_group.items():
#     print(f"Instance Group: {group_name}")
#     print("Top 25 TF-IDF Words (Document Frequency Normalized):")
#     print(tfidf_data['tfidf words'])
#     print("\n")
    
#     # Extract terms and their TF-IDF scores
#     terms = tfidf_data['tfidf words'].index
#     scores = tfidf_data['tfidf words'].values
    
#     # Create a DataFrame for plotting
#     tfidf_df = pd.DataFrame({'Term': terms, 'TF-IDF Score': scores})
    
#     # Plot
#     plt.figure(figsize=(12, 8))
#     plt.barh(tfidf_df['Term'], tfidf_df['TF-IDF Score'], color='skyblue')
#     plt.xlabel('TF-IDF Score')
#     plt.ylabel('Terms')
#     plt.title(f'TF-IDF Scores for {group_name} (Document Frequency Normalized)')
#     plt.gca().invert_yaxis()  # Highest scores on top
#     plt.show()

# # Group by 'instance group'
# groups = data.groupby('instance group')

# # Initialize a dictionary to store TF-IDF for each group
# tfidf_per_group = {}
# all_top_words = pd.Series(dtype=float)

# # Loop over each group to compute the TF-IDF and accumulate top words
# for group_name, group_data in groups:
#     # Get the 'rules' column for the current group
#     rules_text = group_data['cleaned translated text'].dropna().values
    
#     # Initialize TF-IDF vectorizer
#     tfidf = TfidfVectorizer()
    
#     # Fit and transform the 'rules' column into a TF-IDF matrix
#     tfidf_matrix = tfidf.fit_transform(rules_text)
    
#     # Convert the matrix to a DataFrame
#     tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    
#     # Normalize by the number of documents in the group (document frequency normalization)
#     num_docs = len(rules_text)
#     normalized_tfidf_df = tfidf_df / num_docs
    
#     # Sum TF-IDF scores and get top 25 words
#     top_tfidf_words = normalized_tfidf_df.sum().sort_values(ascending=False).head(25)
    
#     # Store the top words in a cumulative series
#     all_top_words = all_top_words.add(top_tfidf_words, fill_value=0)
    
#     # Store the normalized TF-IDF for this group
#     tfidf_per_group[group_name] = normalized_tfidf_df

# # Get the unified set of top 25 words across all groups
# unified_top_words = all_top_words.sort_values(ascending=False).head(25).index

# # Loop through each group again to align the TF-IDF DataFrames with the unified top words
# for group_name, normalized_tfidf_df in tfidf_per_group.items():
#     # Reindex the DataFrame to only include the unified top 25 words
#     reindexed_tfidf_df = normalized_tfidf_df.reindex(columns=unified_top_words, fill_value=0)
    
#     # Sum the TF-IDF scores for plotting
#     top_tfidf_scores = reindexed_tfidf_df.sum()

#     # Plot the TF-IDF scores of the unified top 25 words for this group
#     plt.figure(figsize=(12, 8))
#     plt.barh(unified_top_words, top_tfidf_scores, color='skyblue')
#     plt.xlabel('TF-IDF Score')
#     plt.ylabel('Terms')
#     plt.title(f'TF-IDF Scores for {group_name} (Unified Top 25 Words)')
#     plt.gca().invert_yaxis()  # Highest scores on top
#     plt.show()

#     print(f"Group: {group_name}")
#     print(f"Unified TF-IDF Top Words (Document Frequency Normalized):")
#     print(top_tfidf_scores)
#     print("\n")

# Group by 'instance_group'
groups = data.groupby('instance group')

# Initialize dictionaries to store TF-IDF, top words, and group sizes
tfidf_per_group = {}
top_words_per_group = []
group_sizes = {}

# Step 1: Loop over each group, compute the TF-IDF, and collect the top 25 words for each group
for group_name, group_data in groups:
    # Get the 'rules' column for the current group
    rules_text = group_data['cleaned translated text'].dropna().values
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    
    # Fit and transform the 'rules' column into a TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(rules_text)
    
    # Convert the matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    
    # Normalize by the number of documents in the group (document frequency normalization)
    num_docs = len(rules_text)
    normalized_tfidf_df = tfidf_df / num_docs
    
    # Sum TF-IDF scores and get top 25 words
    top_tfidf_words = normalized_tfidf_df.sum().sort_values(ascending=False).head(25)
    
    # Store the top words
    top_words_per_group.append(top_tfidf_words.index)
    
    # Store the normalized TF-IDF and group size
    tfidf_per_group[group_name] = {
        'tfidf_matrix': normalized_tfidf_df,
        'tfidf words': top_tfidf_words
    }
    group_sizes[group_name] = num_docs

# Step 2: Create a unified set of top words across all groups
unified_top_words = pd.Index(set().union(*top_words_per_group))  # Unified set of top 25 words across all groups

# Step 3: Recompute TF-IDF for each group using the unified set of top words and plot the results
for group_name, tfidf_data in tfidf_per_group.items():
    normalized_tfidf_df = tfidf_data['tfidf_matrix']
    
    # Reindex the TF-IDF DataFrame to include only the unified top words (fill missing words with 0)
    unified_tfidf_df = normalized_tfidf_df.reindex(columns=unified_top_words, fill_value=0)
    
    # Sum TF-IDF scores for each word
    top_tfidf_words = unified_tfidf_df.sum().sort_values(ascending=False)
    
    # Store the unified TF-IDF matrix for this group
    tfidf_per_group[group_name]['unified_tfidf'] = top_tfidf_words
    
    # Extract terms and their TF-IDF scores
    terms = top_tfidf_words.index
    scores = top_tfidf_words.values
    
    # Create a DataFrame for plotting
    tfidf_df = pd.DataFrame({'Term': terms, 'TF-IDF Score': scores})
    
    # # Plot
    # plt.figure(figsize=(12, 8))
    # plt.barh(tfidf_df['Term'], tfidf_df['TF-IDF Score'], color='skyblue')
    # plt.xlabel('TF-IDF Score')
    # plt.ylabel('Terms')
    # plt.title(f'TF-IDF Scores for {group_name} (Unified Top 25 Words)')
    # plt.gca().invert_yaxis()  # Highest scores on top
    # plt.show()

# Define your specific order for group names
bins = ['1 to 5', '6 to 15', '16 to 50', '51 to 150', '151 to 500', '501 to 1500', '1501 to 5000', '5001+']

# Prepare data for plotting
plot_data = []

for word in unified_top_words[:6]:
    scores = []
    group_names = []
    
    for group_name in bins:
        if group_name in tfidf_per_group:
            tfidf_scores = tfidf_per_group[group_name]['unified_tfidf']
            group_names.append(group_name)
            scores.append(tfidf_scores.get(word, 0))
        else:
            # If a group is missing in the data, handle it (e.g., append zero score)
            group_names.append(group_name)
            scores.append(0)
    
    # Create a DataFrame for each word
    temp_df = pd.DataFrame({
        'Group Name': group_names,
        'TF-IDF Score': scores,
        'Word': word
    })
    
    plot_data.append(temp_df)

# Concatenate all DataFrames into a single DataFrame for plotting
plot_df = pd.concat(plot_data)

# Plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=plot_df, y='TF-IDF Score', x='Group Name', hue='Word', marker='o', linestyle='-', palette='tab10', hue_order=unified_top_words)
plt.xlabel('TF-IDF Score')
plt.ylabel('Group Name')
plt.title('TF-IDF Scores Across Groups for Unified Top 25 Words')
plt.legend(title='Word')
plt.grid(True)
plt.show()