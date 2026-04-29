# robustness_rule_deduplicate_llm_categorization.py
#
# Applies GPT-4o-mini one-shot categorization to the deduplicated rule set to
# verify that deduplication does not substantially change the topic distribution.
# Identical prompt and category taxonomy as the main LLM categorization script.
#
# Input:  data/mastodon_rules_deduplicated.csv  (deduplicated rules from robustness_rules_deduplicate.py)
# Output: data/categorized_my_rules_gpt4omini_full.csv

import os
import pandas as pd
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.environ.get('OPEN_AI_KEY')

# Load deduplicated rules; drop rows with missing or single-word translations
df = pd.read_csv(r'data\deduplicate\mastodon_rules_deduplicated.csv')
df = df.dropna(subset=['translated text'])
df = df[df['translated text'].str.split().str.len() >= 1]
df = df.rename(columns={'instance': 'Instance Name'})

# Exclude instances with atypically long or legally sensitive rule sets
df = df[~df['Instance Name'].isin(['libranet.de', 'venera.social', 'social.outsourcedmath.com'])]

gpt_model = "gpt-4o-mini"

rule_types = [
    'Advertising & Commercialization', 'Copyright/ Piracy',
    'Doxxing/ Personal Info', 'Harassment', 'Hate Speech', 'Images',
    'Links & Outside Content', 'NSFW', 'Off-topic/topic specific', 'Dogpiling',
    'Reposting/Crossposting', 'Spam', 'Trolling', 'Incitement of Violence',
    'Mis/Disinformation/Conspiracy', 'Illegal Content', 'Content Warnings',
    'Impersonation', 'Automated tools', 'Not Applicable'
]

prompt = f"""
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

results = []
for rule in df['translated text']:
    response = openai.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f'Rule: "{rule}"'}
        ],
        temperature=0,
        top_p=0
    )
    results.append(response.choices[0].message.content.split("\n"))

df['GPT category'] = results
df.to_csv(r'data\categorized_my_rules_gpt4omini_full.csv', index=False)
