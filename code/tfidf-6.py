#plotly with ranks 

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objs as go
import numpy as np
import html
import regex as re
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

# Step 1: Calculate the global TF-IDF for the entire dataset
tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(data['cleaned translated text'].dropna())
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
num_docs = tfidf_matrix.shape[0]



# Step 3: Sum the Tf-idf score across all documents to find the most dominant docs 
top_tfidf_words = tfidf_df.sum().sort_values(ascending=False)
top_words = top_tfidf_words.index
print("Top Words across entire dataset:", top_words)

# Dictionary mapping words to topics
word_to_topic = {
    'racism': 'hate speech',
    'homophobia': 'hate speech',
    'transphobia': 'hate speech',
    'xenophobia': 'hate speech',
    'hate': 'hate speech', 
    'sexism': 'hate speech',
    'casteism' : 'hate speech',
    'discrimination' : 'hate speech',
    'violent': 'violence', 
    'violence': 'violence', 
    'incitement' : 'violence',
    'kind': 'general respectfulness', 
    'nice': 'general respectfulness', 
    'respectful': 'general respectfulness', 
    'respect' : 'general respectfulness',
    'instance': 'mastodon entities', 
    'good' : 'general respectfulness',
    'account': 'mastodon entities', 
    'accounts': 'mastodon entities', 
    'posts': 'mastodon entities',
    'server' : 'mastodon entities', 
    'post' : 'mastodon entities', 
    'users' : 'mastodon entities', 
    'user' : 'mastodon entities',
    'mastodon' : 'mastodon entities',
    'community' : 'mastodon entities',
    'harassment' : 'targetted content', 
    'dogpiling' : 'targetted content', 
    'sensitive' : 'sexually sensitive', 
    'nsfw' : 'sexually sensitive', 
    'explicit' : 'sexually sensitive', 
    'pornography' : 'sexually sensitive',
    'sexually' : 'sexually sensitive', 
    'sexual' : 'sexually sensitive',
    'marked' : 'sexually sensitive', 
    'cw' : 'sexually sensitive', 
    'warning' : 'sexually sensitive',
    'warnings' : 'sexually sensitive',
    'posting' :'action', 
    'share' : 'action', 
    'publish' : 'action',
    'false' : 'misinformation', 
    'misleading' : 'misinformation', 
    'intentionally' : 'misinformation',
    'dont' : 'restrictive', 
    'prohibited' : 'restrictive', 
    'forbidden' : 'restrictive', 
    'avoid' : 'restrictive',
    'illegal' : 'legal', 
    'law' : 'legal', 
    'laws' : 'legal', 
    'allowed' : 'prescriptive', 
    'permitted' : 'prescriptive', 
    'follow' : 'prescriptive', 
    'conduct' : 'prescriptive', 
    'media' : 'media', 
    'links' : 'media', 
    'images' : 'media', 
    'bots' : 'bots', 
    'spam': 'spam', 
    'dogpiling' : 'dogpiling'
}

words_to_remove = ['information', 'rule', 'rules', 'people', 
                   'personal', 'use', 'right', 'make', 'language', 
                   'includes', 'service', 'data', 'person', 'like', 'want', 
                   'germany', 'used', 'avoid', 'form', 'just', 'site', 'does', 'privacy',
                   'links', 'limited', 'local', 'dictionary', 'used', 'follow', 'welcome', 'andor', 'list', 'gender', 
                   'action', 'private', 'public', 'including', 'speech']

def remove_unnecessary_words(df, words_to_remove):
    """
    Remove specified words (columns) from the DataFrame.
    
    :param df: DataFrame from which to remove words.
    :param words_to_remove: List of words (columns) to remove.
    """
    # Drop the columns that are in words_to_remove and also exist in the DataFrame
    df.drop(columns=[word for word in words_to_remove if word in df.columns], inplace=True)



# Group by 'instance_group'
groups = data.groupby('instance group')


# Initialize DataFrame for ranked scores, including both topic and non-topic words
ranked_scores = pd.DataFrame(columns=top_words)

# Get ranks for each word in each group
for group_name, group_data in groups:
    # Get the 'cleaned translated text' column for the current group
    rules_text = group_data['cleaned translated text'].dropna().values
    
    # Initialize the TF-IDF vectorizer
    tfidf = TfidfVectorizer(vocabulary=top_words)
    
    # Fit and transform the 'rules' column into a TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(rules_text)
    
    # Convert the matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # Get feature names (which will be the same as top_words)
    words = tfidf.get_feature_names_out()
    
    # Compute the sum of TF-IDF scores for ranking
    summed_scores = tfidf_df.sum().sort_values(ascending=False)
    
    # Get ranks based on the summed TF-IDF scores
    ranks = summed_scores.rank(ascending=False)
    
    # Add ranks to the ranked_scores DataFrame
    ranked_scores.loc[group_name] = ranks

# Retain all words (topic and non-topic)
rescored_ranks = pd.DataFrame(index=ranked_scores.index)

# Include topic-based ranks
for word, topic in word_to_topic.items():
    if word in ranked_scores.columns and word not in words_to_remove:
        rescored_ranks[topic] = ranked_scores[word]

# Include non-topic words (that are not in words_to_remove)
non_topic_words = [word for word in ranked_scores.columns if word not in word_to_topic and word not in words_to_remove]
for non_topic in non_topic_words:
    rescored_ranks[non_topic] = ranked_scores[non_topic]

# # Normalize order for specific instance groups
# custom_order = ['1 to 5', '6 to 15', '16 to 50', '51 to 150', '151 to 500', '501 to 1500', '1501 to 5000', '5001+']
# rescored_ranks = rescored_ranks.loc[custom_order]

# # List of categories for the plot
# categories = rescored_ranks.columns  # Categories are the columns
# print(categories)

# # Create traces for each category
# traces = []
# for category in categories:
#     traces.append(
#         go.Scatter(
#             x=rescored_ranks.index,  # Words as x-axis
#             y=rescored_ranks[category],  # Ranks for each category
#             mode='lines+markers',  # Display as lines with markers
#             name=category,  # Name of the category for the legend
#             line_shape='spline'  # Smooth lines
#         )
#     )

# # Create the layout for the plot
# layout = go.Layout(
#     title="Word Ranks Across Categories",
#     xaxis={'title': 'Instance Groups'},
#     yaxis={'title': 'Ranks', 'autorange': 'reversed'},  # Reverse the y-axis
#     hovermode='closest'
# )

# # Create the figure and plot
# fig = go.Figure(data=traces, layout=layout)
# fig.show()

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Normalize order for specific instance groups
custom_order = ['1 to 5', '6 to 15', '16 to 50', '51 to 150', '151 to 500', '501 to 1500', '1501 to 5000', '5001+']
# rescored_ranks = rescored_ranks.loc[custom_order]

# # Define words for each panel
# panel_words = {
#     "Restrictive & Prescriptive": ['restrictive', 'prescriptive'],
#     "Behavior": ['targetted content', 'dogpiling', 'general respectfulness'],
#     "Content" : ['legal', 'hate speech', 'violence', 'sexual content', 'media', 'misinformation'],
#     "Distribution": ['spam', 'ads and promotion', 'reposting'],
#     "Action" : ['bots']
# }

# # Create a 2x3 subplot grid (5 panels)
# fig = make_subplots(rows=2, cols=3, 
#                     subplot_titles=list(panel_words.keys()),
#                     vertical_spacing=0.2, horizontal_spacing=0.1)

# # Colors to differentiate the traces
# colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']

# # Create traces for each panel
# for i, (panel_title, words) in enumerate(panel_words.items()):
#     row = (i // 3) + 1  # Calculate row number (1-based)
#     col = (i % 3) + 1  # Calculate column number (1-based)
    
#     for j, word in enumerate(words):
#         if word in rescored_ranks.columns:
#             fig.add_trace(
#                 go.Scatter(
#                     x=rescored_ranks.index,  # Instance groups as x-axis
#                     y=rescored_ranks[word],  # Ranks for each word
#                     mode='lines+markers',  # Display as lines with markers
#                     name=f"{panel_title}: {word}",  # Unique name combining panel and word for legend
#                     line_shape='spline',  # Smooth lines
#                     marker=dict(color=colors[j % len(colors)])  # Cycle through colors
#                 ),
#                 row=row, col=col  # Specify the row and column for the trace
#             )

# # Update layout
# fig.update_layout(
#     title="Word Ranks Across Different Categories",
#     height=800, width=1200,
#     showlegend=True,  # Enable global legend
#     legend=dict(
#         title="Legend",
#         x=1.05,  # Position legend outside of plot area
#         y=0.5,
#         traceorder='normal',
#         font=dict(size=10),
#         orientation='v'
#     )
# )

# # Update x and y axes for each subplot
# fig.update_xaxes(title_text="Instance Groups", row=2, col=1)
# fig.update_yaxes(title_text="Ranks", autorange="reversed")

# # Show the figure
# fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots



rescored_ranks = rescored_ranks.loc[custom_order]

# Define the categories and their respective words
panel_words = {
    "Restrictive": ['restrictive'],
    "Prescriptive": ['prescriptive'],
    "Categories Avg": {
        "Behavior": ['targetted content', 'dogpiling', 'general respectfulness'],
        "Content": ['legal', 'hate speech', 'violence', 'sexually sensitive', 'media', 'misinformation'],
        "Distribution": ['spam'],
        "Action": ['bots']
    }
}

# Create a 1x3 subplot grid (3 panels)
fig = make_subplots(rows=1, cols=3, 
                    subplot_titles=["Restrictive", "Prescriptive", "Action, Behavior, Content, Distribution Avg"],
                    vertical_spacing=0.2, horizontal_spacing=0.1)

# Colors to differentiate the traces
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']  # One color per category

# Plot for "Restrictive" in the first panel
if 'restrictive' in rescored_ranks.columns:
    fig.add_trace(
        go.Scatter(
            x=rescored_ranks.index,
            y=rescored_ranks['restrictive'],
            mode='lines+markers',
            name='Restrictive',
            line_shape='spline',
            marker=dict(color=colors[0])
        ),
        row=1, col=1
    )

# Plot for "Prescriptive" in the second panel
if 'prescriptive' in rescored_ranks.columns:
    fig.add_trace(
        go.Scatter(
            x=rescored_ranks.index,
            y=rescored_ranks['prescriptive'],
            mode='lines+markers',
            name='Prescriptive',
            line_shape='spline',
            marker=dict(color=colors[1])
        ),
        row=1, col=2
    )

# Plot average scores for each category in the third panel
for i, (category, words) in enumerate(panel_words["Categories Avg"].items()):
    # Calculate the average rank for the words in the current category
    category_avg = rescored_ranks[words].mean(axis=1)
    
    fig.add_trace(
        go.Scatter(
            x=rescored_ranks.index,
            y=category_avg,
            mode='lines+markers',
            name=category,  # Category name for the legend
            line_shape='spline',
            marker=dict(color=colors[i])  # Assign a color for each category
        ),
        row=1, col=3
    )

# Update layout
fig.update_layout(
    title="Word Ranks Across Different Categories",
    height=600, width=1200,
    showlegend=True,  # Enable global legend
    legend=dict(
        title="Legend",
        x=1.05,  # Position legend outside of plot area
        y=0.5,
        traceorder='normal',
        font=dict(size=10),
        orientation='v'
    )
)

# Update x and y axes for each subplot
fig.update_xaxes(title_text="Instance Groups", row=1, col=1)
fig.update_xaxes(title_text="Instance Groups", row=1, col=2)
fig.update_xaxes(title_text="Instance Groups", row=1, col=3)
fig.update_yaxes(title_text="Ranks", autorange="reversed")

# Show the figure
fig.show()
