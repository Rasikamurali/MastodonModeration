"""
llm_test_mastodon_ucb.py

This script uses OpenAI's GPT model to categorize rules from a consolidated annotated Mastodon dataset into predefined rule types.
- Loads rules from 'Consolidated_annotated_ucb_dataset.csv'.
- Sends each rule to the OpenAI API for categorization.
- Saves the categorized results to 'sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv'.
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
import random
import re

# Set the seed to 42
random.seed(42)


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the consolidated annotated Mastodon dataset
# (Make sure the file path is correct and the file exists)
data = pd.read_csv(r'Consolidated_annotated_ucb_dataset.csv')
print(len(data))

print(data.columns)


gpt_model = "gpt-4o-mini"


rule_types = ['Prescriptive', 'Restrictive', 'Behavior/Content/Format Allowed',
       'Consequences/Moderation/Enforcement',
       'Advertising & Commercialization', 'Copyright/ Piracy',
       'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
       'Links & Outside Content', 'Low-Quality Content', 'NSFW',
       'Off-topic/topic specific', 'Personal Army/Brigading/Dogpiling',
       'Personality', 'Politics', 'Reddiquette/ Sitewide',
       'Reposting/Crossposting', 'Spam', 'Spoilers', 'Trolling',
       'Incitement of Violence', 'Mis/Disinformation/Conspiracy',
       'Illegal Content', 'No CSAM', 'Content Warnings', 'Gore/Violence',
       'Animal abuse', 'Impersonation', 'Server Interference',
       'Other form of oppresive -ism', 'Automated tools',
       'English Language Only', 'Moderator power', 'Age Restriction',
       'Privacy Norms', 'NFT/Crypto', 'AI generated content',
       'Codes of Conduct', 'Defederation', 'Sign Up']


# Prepare the prompt for the OpenAI model
# (This prompt instructs the model to categorize rules into the specified types)
prompt1 = f"""
# Q: Based on the given {rule_types}, select the appropriate categories for the rule: Sexually explicit or violent media must be marked as sensitive when posting. 
# A: Prescriptive, Images

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: No spam or advertising 
# A: Restrictive, Spam

# Q: Based on the given {rule_types}, select the approriate categories for the rule: People registering and maintaining accounts at Romancelandia.Club agree to these rules as the starting point for conduct guidance. You are also agreeing to review and abide by the full code of conduct (below).
# A: Codes of Conduct, Sign Up

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: No racism, sexism, homophobia, transphobia, xenophobia, or casteism.
# A: Restrictive, Hate Speech, Reddiquette 

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: This server abides by the Mastodon Covenant
# A: Prescriptive, Reddiquette

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: No incitement of violence or promotion of violent ideologies.
# A: Restrictive, Personal Army, Incitement of Violence 

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: No harassment, dogpiling or doxing of other.
# A: Restrictive, Doxxing/Personal Info, Harassment, Personal Army

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: Hywel is running this instance out of his own pocket for BDSM models, producers and fans. If you don't like the way this instance is run, start your own. That's what Mastodon is for :-)
# A: Server Interference

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: Respect other users' boundaries.
# A: Prescriptive, Personality 

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: No trolling, microaggressions, or harassment. See the Code of Conduct for the full list.
# A: Restrictive, Harassment, Trolling, Codes of Conduct 

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: Conduct intended to stalk or harass users, impede users from using the service, degrade the performance of the service, or incite others to perform any of the aforementioned actions, is disallowed. 
# A: Prescriptive, Copyright/Piracy, Harassment, Server Interference

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: No NFT content. None at all. There are crypto-related instances on the fediverse that would be more suitable.
# A: Restrictive, NFT/Crypto

# Q: Based on the given {rule_types}, select the appropriate categories for the rule: Controversial, reactionary, NSFW, and satirical or similar content may be removed if not obviously Content Warned where appropriate.
# A: Consequences/Moderation/Enforcement, NSFW, Content Warnings

Based on the given {rule_types}, select the appropriate categories for the rule:
"""

language_assessment = []
rules = data['rules_list']
for rule in rules:
  response = openai.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": prompt1},
        {"role": "user", "content": f'Rule: {rule}'}
        ]
  )
      # The model's response is split into lines for easier processing
  language_assessment.append(response.choices[0].message.content.split("\n"))

data['GPT category'] = language_assessment
print(data.head())
data.to_csv('sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv')


