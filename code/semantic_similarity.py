from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Sentence-BERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

content_df = pd.read_csv('content.csv')
about_rules = pd.read_csv('about_rules.csv')

#print(content_df.head())
#print(about_rules.head())

names = about_rules['instance']
instance_name = []
for name in names: 
    naming = (name.split('.'))
    instance_name.append(naming[0] + naming[1])

about_rules['instance name'] = instance_name
#print(about_rules.head())

for name in ['mastodoncloud', 'mastodonsocial']: 
    print(name)
    filtered_rows = content_df[content_df['instance name']==name]
    final_toots = filtered_rows['toot_content']
    final_toots.to_csv(f'{name}_content.csv')
    # filtered_rules = about_rules[about_rules['instance name']==name]
    # final_rules = filtered_rules['rules']
    # content_embedding = model.encode(final_toots)
    # rules_embedding = model.encode(final_rules)
    # similarity_score = cosine_similarity(content_embedding, rules_embedding)[0][0]
    # print(similarity_score)

for name in ['mastodoncloud', 'mastodonsocial']: 
    print(name)
    content = pd.read_csv(rf'{name}_content.csv')
    toots = content['toot_content']
    filtered_rules = about_rules[about_rules['instance name']==name]
    final_rules = filtered_rules['rules']
    content_embedding = model.encode(toots)
    rules_embedding = model.encode(final_rules)
    similarity_score = cosine_similarity(content_embedding, rules_embedding)[0][0]
    print(similarity_score)



# # Function to calculate semantic similarity between content and rules
# def calculate_similarity(content, rules):
#     content_embedding = model.encode([content])
#     rules_embedding = model.encode([rules])
#     similarity_score = cosine_similarity(content_embedding, rules_embedding)[0][0]
#     return similarity_score

# # Iterate through each row of the DataFrame
# for index, row in df.iterrows():
#     instance_name = row['instance name']
#     content = row['content']
#     rules = row['rules']
    
#     # Calculate semantic similarity between content and rules
#     similarity_score = calculate_similarity(content, rules)
    
#     # Determine if the similarity score meets a predefined threshold
#     threshold = 0.8  # Adjust as needed
#     if similarity_score >= threshold:
#         print(f"Content '{content}' meets the rules for instance '{instance_name}'")
#     else:
#         print(f"Content '{content}' does not meet the rules for instance '{instance_name}'")
