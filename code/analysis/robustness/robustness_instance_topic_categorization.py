# robustness_instance_topic_categorization.py
#
# Uses GPT-4o-mini with one-shot prompting to classify Mastodon instance short
# descriptions into one of 26 topic categories (e.g. Gaming, LGBTQ, Tech).
# Processes instances in batches (script shows rows 5000-6000); results are
# used to assess whether instance topic predicts rule content patterns.
#
# Input:  data/instance_topics_translated.csv  (translated short descriptions, full set)
#         data/sampled_instances_topics.csv    (200-instance annotation sample)
# Output: data/sampled_annotations_GPT_category(4)_5000_6000.csv

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import openai
from collections import Counter
from functools import reduce
from dotenv import load_dotenv
import time
import os


load_dotenv()
openai.api_key = os.environ.get('OPEN_AI_KEY')
# Load translated instance descriptions for GPT categorization
df = pd.read_csv(r'data\instance_topics_translated.csv')

print(df.columns)
print(len(df))

data = df.dropna(subset='translated short')
print(len(data))

df2 = pd.read_csv(r'data\sampled_instances_topics.csv')
print(len(df2))
df2 = df2[['Instance Name', 'translated short']].drop_duplicates()
print(len(df2))


gpt_model = "gpt-4o-mini"

rule_types = [
  'General', 'Family & Friends', 'Personal', 'Gaming', 'Tech', 'Fediverse', 'Geographic', 'Interests', 'Ideology', 
  'Entertainment', 'Interests', 'Ideology', 'Sexual Content', 'Art & Culture', 'News', 'LGBTQ', 'Self-improvement', 'Activism',
    'Organization', 'Support', 'Politics', 'Bots', 'Religion', 'Pets', 'N/A', 'Memes'
       ]

prompt1 = f"""

# You are given a list of topic types. Each topic type has a description that defines the type of instance it applies to. 
# Use the following categories and their descriptions to classify instances:

# # General: Instances that range in a wide variety of topics

# # Family & Friends: Instances that explicitly mention that it is only for friends and family; or invite-only for people in the know

# # Personal: Instances that explicitly mention that it is a personal instance

# # Gaming: Instances that are about video games

# # Tech: Instances about various tech topics such as coding, programming, open source etc. etc.

# # Fediverse: Instances are coming from other Fediverse platforms such as Pleroma or Pixelfed

# # Geographic: Instances that are specifically for people of some countries or specific cities

# # Interests: Instances that are for niche interests that may not fit into other categories

# # Ideology: Instances that specify focus on ideologies (such as equality, freedom, etc.)

# # Entertainment: Instances that are about books, TV, shows, fandoms, music etc.

# # Sexual content: Instances that are about sexual material including but not limited to photo-sharing and sexual role-play

# # LGBTQ: Instances that explicitly state they are for people of the LGBTQ community

# # News: Instances that focus on conversations about news

# # Art & Culture: Instances about the creation and sharing of art & culture

# # Self-improvement: Instances for people to talk about ways to improve themselves and the world around them

# # Activism: Instances for promoting activism related to more politically charged topics

# # Organization: Instances for organizations such as companies, schools/universities, club chapter

# # Support: Instances that focus on safety and inclusivity

# # Politics: Instances to talk about politics and current affairs

# # Bots: Instances specifically for bots

# # Religion: Instance focused on conversation about faith

# # Pets: Instances to talk about all sorts of pets such as dogs, cats etc.

# # N/A: Instances that don’t specify purpose or target audience

# # Memes: Instances about the sharing of memes and deep discussion about its cultural relevance

# Examples Tasks:
# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "Pocket friends are welcome."
# A: N/A

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "George G's Mastodon for the personal use of its users.." 
# A: Personal

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "A Mastodon instance emanating from Waterford city in Ireland. Mastodon is a decentralised, federated microblogging social network."
# A: Tech

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "Pixelfed instance dedicated to trains.."
# A: Fediverse

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "A Mastodon server run by Botany One on behalf of the Annals of Botany Company, a UK charity set up to promote the study of plants."
# A: Organization

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "Splatoon Series This is a mastodon server for Korean fans.Welcome to all those who love the Splatoon series!"
# A: Gaming

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "For everyone interested in wikis, editor or not! Wikipedia, Bulbapedia, independent wikis, anything."
# A: Interests

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "Your cozy & queer gamedev community is hereâœ¨."
# A: LGBTQ, Gaming 

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "I'll take a look around"
# A: N/A

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "Innovators Welcome! Every month we meet new creators and help develop new projects. Our Concept is simple, provide assistance to help creators develop and get into our industry."
# A: Support  

# Q: Based on the given {rule_types}, select the appropriate categories for the instance description: "This is a server for incest (and kink) posting. This server treats consensual incest in a serious and accepting way. We are open to incest of any kind as long as it is consensual and between adults." 
# A: Sexual Content



Based on the given {rule_types}, select the appropriate categories for the instance description:
"""


data = data[5000:]
language_assessment = []
rules = data['translated short']

for rule in rules:
  #print(rule)
  response = openai.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": prompt1},
        {"role": "user", "content": f'Instance description: "{rule}"'}
        ], 
    temperature=0,  
    top_p=0 
  )
  language_assessment.append(response.choices[0].message.content.split("\n"))
  
# print(language_assessment)
data['GPT category'] = language_assessment
# print(data.head())
data.to_csv('data\sampled_annotations_GPT_category(4)_5000_6000.csv')

