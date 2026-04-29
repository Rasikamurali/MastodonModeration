import pandas as pd 
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import html
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


data = pd.read_csv(r'Independent Study\data\translated_rules_dataset.csv')
data = data.dropna(subset=['translated text'])

# Cleaning function
def clean_text(text_list):
    cleaned_list = []
    for text in text_list:
        text = text.lower()  # Lowercase the text
        text = html.unescape(text)  # Remove HTML entities
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation using regex
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
        cleaned_text = ' '.join(tokens)  # Join the cleaned tokens back into a single string
        cleaned_list.append(cleaned_text)
    return cleaned_list

# Apply cleaning to data
data['cleaned translated text'] = clean_text(data['translated text'])
print(data.head())

# Group by 'instance_group'
groups = data.groupby('instance group')

# Initialize dictionaries to store TF-IDF, top words, and group sizes
tfidf_per_group = {}
top_words_per_group = []
group_sizes = {}

# Step 1: Loop over each group, compute the TF-IDF, and collect the top 25 words for each group
for group_name, group_data in groups:
    rules_text = group_data['cleaned translated text'].dropna().values
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(rules_text)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    num_docs = len(rules_text)
    normalized_tfidf_df = tfidf_df / num_docs
    top_tfidf_words = normalized_tfidf_df.sum().sort_values(ascending=False).head(25)
    top_words_per_group.append(top_tfidf_words.index)
    tfidf_per_group[group_name] = {
        'tfidf_matrix': normalized_tfidf_df,
        'tfidf words': top_tfidf_words
    }
    group_sizes[group_name] = num_docs

# Step 2: Create a unified set of top words across all groups
unified_top_words = pd.Index(set().union(*top_words_per_group))

# Step 3: Recompute TF-IDF for each group using the unified set of top words
for group_name, tfidf_data in tfidf_per_group.items():
    normalized_tfidf_df = tfidf_data['tfidf_matrix']
    unified_tfidf_df = normalized_tfidf_df.reindex(columns=unified_top_words, fill_value=0)
    top_tfidf_words = unified_tfidf_df.sum().sort_values(ascending=False)
    tfidf_per_group[group_name]['unified_tfidf'] = top_tfidf_words
    terms = top_tfidf_words.index
    scores = top_tfidf_words.values
    tfidf_df = pd.DataFrame({'Term': terms, 'TF-IDF Score': scores})

# Define specific order for group names
bins = ['1 to 5', '6 to 15', '16 to 50', '51 to 150', '151 to 500', '501 to 1500', '1501 to 5000', '5001+']

# Removing certain words from the set
def remove_unwanted_words(tfidf_df):
    return tfidf_df[~tfidf_df['Term'].isin(['use', 'rule', 'rules', 'list', 'dictionary'])]

# Dictionary mapping words to topics
word_to_topic = {
    'racism': 'hate speech',
    'homophobia': 'hate speech',
    'transphobia': 'hate speech',
    'xenophobia': 'hate speech',
    'hate': 'hate speech', 
    'sexism': 'hate speech',
    'violent': 'violence', 
    'violence': 'violence', 
    'kind': 'general respectfulness', 
    'nice': 'general respectfulness', 
    'respect': 'general respectfulness', 
    'instance': 'mastodon entities', 
    'account': 'mastodon entities', 
    'accounts': 'mastodon entities', 
    'posts': 'mastodon entities',
}

# Function to compute the topic-level TF-IDF scores for each group (bin)
def compute_topic_tfidf_per_bin(group_name, tfidf_data):
    # Get the normalized TF-IDF DataFrame
    normalized_tfidf_df = tfidf_data['unified_tfidf']
    
    # Convert to DataFrame for processing
    tfidf_df = pd.DataFrame({
        'Term': normalized_tfidf_df.index,
        'TF-IDF Score': normalized_tfidf_df.values
    })
    
    # Remove unwanted words
    tfidf_df = remove_unwanted_words(tfidf_df)
    
    # Add a column for the topic
    tfidf_df['topic'] = tfidf_df['Term'].map(word_to_topic)
    
    # Separate rows with topics and without topics
    df_with_topics = tfidf_df[tfidf_df['topic'].notna()].copy()
    df_without_topics = tfidf_df[tfidf_df['topic'].isna()].copy()
    
    # Group by 'topic' and calculate the average tf-idf score
    topic_avg_tfidf = df_with_topics.groupby('topic')['TF-IDF Score'].mean().reset_index()
    
    # Rename 'topic' column to 'Term' for consistency with the original dataset
    topic_avg_tfidf = topic_avg_tfidf.rename(columns={'topic': 'Term'})
    
    # Combine the new topic-averaged scores with the words that don't have topics
    result_df = pd.concat([df_without_topics[['Term', 'TF-IDF Score']], topic_avg_tfidf], ignore_index=True)
    
    # Add a 'Group Name' column to result_df for clarity
    result_df['Group Name'] = group_name
    
    return result_df

# Prepare data for plotting and observing patterns across bins
plot_data = []

# Loop through each group (bin) and apply the topic-level TF-IDF computation
for group_name in bins:
    if group_name in tfidf_per_group:
        tfidf_data = tfidf_per_group[group_name]
        # Compute topic-level TF-IDF scores for this group
        result_df = compute_topic_tfidf_per_bin(group_name, tfidf_data)
        # Append the result to the plot_data list
        plot_data.append(result_df)

# Concatenate all the results into a single DataFrame
final_plot_df = pd.concat(plot_data)

# Display the final DataFrame with topic-level TF-IDF scores for each bin
print(final_plot_df)

# Plotting the results for each topic across bins
unique_terms = final_plot_df['Term'].unique()

# Loop through each unique term to create bar plots
for term in unique_terms:
    term_data = final_plot_df[final_plot_df['Term'] == term]
    
    plt.figure(figsize=(8, 5))
    plt.bar(term_data['Group Name'], term_data['TF-IDF Score'], color='skyblue')
    plt.title(f"TF-IDF Score for '{term}' across group sizes")
    plt.xlabel('Group Name')
    plt.ylabel('TF-IDF Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()