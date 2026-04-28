# sampling_instance_topic_annotation.py
#
# Draws a random sample of 200 instances with non-null translated short
# descriptions for manual topic annotation. The sample is used as the ground
# truth set for evaluating GPT-based instance topic categorization.
#
# Input:  data/instance_topics_translated.csv  (instance short descriptions, translated)
# Output: data/sampled_instances_topics.csv    (200-instance annotation sample)

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# Load instance metadata with translated short descriptions
# Stratified sampling of instances based on topics and instance groups
mstdn_topics_wnorms = pd.read_csv(r'data/instance_topics_translated.csv')
print(mstdn_topics_wnorms.columns)

instance_topics = mstdn_topics_wnorms[['Instance Name', 'translated short']].drop_duplicates()
print(len(instance_topics))

# if translated short is empty, remove row
instance_topics = instance_topics[instance_topics['translated short'].notna()]
print(len(instance_topics))

# Perform stratified sampling

sampled_instances = instance_topics.sample(n=200, random_state=42)

sampled_instances.to_csv(r'data/sampled_instances_topics.csv', index=False)
