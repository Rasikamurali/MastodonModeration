# llm_my_mstdn.py
#
# Older batch-processing version of the LLM rule categorization script.
# Uses one-shot prompting with GPT-4o-mini to assign rule topic categories
# from a fixed list of 20 types to each rule in the deduplicated dataset.
# Note: contains a leftover batch slice (data = data[20000:]) from
# multi-run processing; the cleaned version is llm_my_mstdn_dedepulicated.py.
#
# Input:  mastodon_rules_deduplicated.csv  (deduplicated rules dataset)
# Output: sampled_categorized_rules_my_mastodon_gpt4omini_deduplicated_0109(5).csv

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import openai
from collections import Counter
from functools import reduce
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.environ.get('OPEN_AI_KEY')
# Load the deduplicated rules; path points to the within-analysis copy
df = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\code\analysis\mastodon_rules_deduplicated.csv')

print(df.columns)
print(len(df))

df = df.dropna(subset='translated text')
print(len(df))

#Extra cleaning 

#removing rules with only 1 word
df = df[df['translated text'].str.split().str.len() >= 1]

df = df.rename(columns={'instance': 'Instance Name'})

#removing rules from these 3 instances because they are too long and legally 
df = df[~df['Instance Name'].isin(['libranet.de', 'venera.social', 'social.outsourcedmath.com'])]

print(len(df))

gpt_model = "gpt-4o-mini"


#Different one for prescriptive and Restricitve? 
#Just check the rule types again and be certain about it 

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

data = df.copy() 

data = data[20000:]
rules = data['translated text']
for rule in rules:
  #print(rule)
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
#print(language_assessment)
data['GPT category'] = language_assessment
print(data.head())
data.to_csv('sampled_categorized_rules_my_mastodon_gpt4omini_deduplicated_0109(5).csv')
# language_assessment = list(np.concatenate(language_assessment).flat)
# print(language_assessment)
# print(Counter(language_assessment))



