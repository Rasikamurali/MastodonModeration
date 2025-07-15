"""
llm_my_mstdn.py

This script uses OpenAI's GPT model to categorize community rules from a Mastodon dataset into predefined rule types.
- Loads rules from 'data/community_rules_dataset.csv'.
- Sends each rule to the OpenAI API for categorization.
- Saves the categorized results to 'sampled_categorized_rules_my_mastodon_gpt4omini_0326(1).csv'.
- Requires the OpenAI API key to be set in a .env file as OPENAI_API_KEY.
"""
#import required libraries
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns 
import openai 
from collections import Counter
from functools import reduce
from dotenv import load_dotenv
import os



load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


# Load the dataset containing community rules
# (Make sure the file path is correct and the file exists)
df = pd.read_csv(r'data\community_rules_dataset.csv')

print(df.columns)
print(len(df))

data = df.dropna(subset='translated text')
print(len(data))

#Extra cleaning 

#removing rules with only 1 word
df = df[df['translated text'].str.split().str.len() >= 1]

df = df.rename(columns={'instance': 'Instance Name'})

#removing rules from these 3 instances because they are too long and legally 
df = df[~df['Instance Name'].isin(['libranet.de', 'venera.social', 'social.outsourcedmath.com'])]

print(len(df))


gpt_model = "gpt-4o-mini"

rule_types = [
       'Advertising & Commercialization', 'Copyright/ Piracy',
       'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
       'Links & Outside Content', 'NSFW',
       'Off-topic/topic specific', 'Dogpiling',
       'Reposting/Crossposting', 'Spam', 'Trolling',
       'Incitement of Violence', 'Mis/Disinformation/Conspiracy',
       'Illegal Content', 'Content Warnings', 
        'Impersonation', 
       'Automated tools', 'Not Applicable']

prompt1 = f"""
# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No harassment, dogpiling or doxing of other."
# A: Doxxing/Personal Info, Harassment

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Be hot like Lily." 
# A: Not Applicable

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No trolling, microaggressions, or harassment. See the Code of Conduct for the full list."
# A: Harassment, Trolling

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No incitement of violence or promotion of violent ideologies."
# A: Incitement of Violence 

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Conduct intended to stalk or harass users, impede users from using the service, degrade the performance of the service, or incite others to perform any of the aforementioned actions, is disallowed." 
# A: Copyright/Piracy, Harassment

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Respect the laws of Germany and the Isle of Man, including those related to hate speech and defamation."
# A: Hate Speech, Illegal Content

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Account registrations are monitored and accounts considered spam (including commercial advertising, political campaigning/propaganda, duplicate accounts or impersonating legal entities) will be moderated or removed."
# A: Advertising & Commercialization, Spam

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No racism, sexism, homophobia, transphobia, xenophobia, or casteism."
# A: Hate Speech

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Disallowed content:  commercial messages, all SPAM, and anything that could be considered NSFW (e.g. pornography)."
# A: NSFW, Spam

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "No posting or reposting links to otherwise illegal content."
# A: Links & Outside Content, Reposting/Crossposting, Illegal Content

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Sexually explicit or violent media must be marked as sensitive when posting." 
# A: Images

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: "Banana." 
# A: Not Applicable


Based on the given {rule_types}, select the appropriate categories for the rule:
"""

language_assessment = []
rules = data['translated text']
for rule in rules:
  response = openai.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": prompt1},
        {"role": "user", "content": f'Rule: "{rule}"'}
        ], 
    temperature=0,  
    top_p=0 
  )
  language_assessment.append(response.choices[0].message.content.split("\n"))
data['GPT category'] = language_assessment
print(data.head())
data.to_csv('sampled_categorized_rules_my_mastodon_gpt4omini_0326(1).csv')




