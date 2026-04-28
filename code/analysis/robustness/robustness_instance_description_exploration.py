# robustness_instance_description_exploration.py
#
# Explores instance short descriptions by merging instance metadata with
# community rules and counting the frequency of translated descriptions.
# Used to understand the diversity of instance purposes in the dataset.
#
# Input:  data/complete_instance_list.csv       (full instance metadata)
#         data/community_rules_data.csv          (community rules per instance)
#         data/instance_topics_translated.csv    (translated short descriptions)
# Output: printed frequency counts of descriptions (no file output)

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load full instance list and the filtered sample with rules
complete_instance_list = pd.read_csv(r'data\complete_instance_list.csv')
mstdn_instance_list = pd.read_csv(r'data\community_rules_data.csv')

print(complete_instance_list.columns)
print(mstdn_instance_list.columns)

reqd_columns_1 = ['Instance Id', 'Instance Name', 'User Count', 'Description', 'Short', 'topic', 'federates with', 'instance_group']
reqd_columns_2 = ['Instance Name', 'rule', 'instance group', 'lang', 'translated text', 'rule count']

complete_instance_list = complete_instance_list[reqd_columns_1]
mstdn_instance_list = mstdn_instance_list[reqd_columns_2]   
print(len(mstdn_instance_list))
mstdn_topics_wnorms = pd.merge(mstdn_instance_list, complete_instance_list, on='Instance Name', how='left')
print(len(mstdn_topics_wnorms))
print(mstdn_topics_wnorms.columns)
# mstdn_topics_wnorms.to_csv(r'data\mstdn_topics_wnorms.csv', index=False)

#----------------------------------------------------

# Analysis of topical norms
# First, we need to explore the Short descriptions and the topics 

mstdn_translated_shorts = pd.read_csv(r'data/instance_topics_translated.csv')

#Let's remove duplicates first
mstdn_translated_shorts = mstdn_translated_shorts.drop_duplicates(subset=['Instance Name'])
print(len(mstdn_translated_shorts))

#Now let's see how many instances have same topics 

# topic_counts = Counter(mstdn_translated_shorts['topic'])
# print(topic_counts.most_common(20))

short_counts = Counter(mstdn_translated_shorts['translated short'])
print(short_counts.most_common(40))
