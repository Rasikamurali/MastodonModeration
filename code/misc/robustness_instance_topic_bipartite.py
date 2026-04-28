# robustness_instance_topic_bipartite.py
#
# Builds a bipartite network linking GPT-assigned instance topic categories
# (e.g. Tech, Gaming) to rule topic categories (e.g. Hate Speech, Spam).
# Computes projections onto each node type and exports .gexf files for
# visualization in Gephi. Duplicate of the main topical_bipartite_analysis.py.
#
# Input:  data/sampled_annotations_GPT_category(4)_full.csv  (GPT instance topics)
#         data/llm_category_analysis_data.csv                 (GPT rule topics per instance)
# Output: bipartite_gpt_rule_topics.gexf  (full bipartite graph)
#         gpt_category_projection.gexf    (instance-topic projection)
#         rule_topic_projection.gexf      (rule-topic projection)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import regex as re
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
import re

# Load GPT-assigned instance topic annotations
topical_df = pd.read_csv(r'data\sampled_annotations_GPT_category(4)_full.csv')    
print(topical_df.columns)
print(len(topical_df))
print(topical_df.head())

GPT_category1 = []

#Cleaning the GPT category column to remove unnecessary characters
for row in topical_df['GPT category']: 
    GPT_category1.append(re.sub(r'A:\s*|[\[\]\'"]', '', row).strip())
    
topical_df['GPT category cleaned'] = GPT_category1

print(topical_df.head())

topical_df = topical_df.drop(columns=['Unnamed: 0'])

rule_topic_data = pd.read_csv(r'data\llm_category_analysis_data.csv')

print(rule_topic_data.columns)
print(len(rule_topic_data))

combined_data = pd.merge(rule_topic_data, topical_df, on='Instance Name', how='inner')
print(len(combined_data))
print(combined_data.columns)    
print(combined_data.head())

######### 
"""
First, group the data by Instance Name. Get set of the rule topics. 
Then, group the data by topic categories assigned by GPT. 
Topic -> Instanceas -> Rule topic sets

"""

# Step 1: Split categories on commas
topical_df['GPT category split'] = topical_df['GPT category cleaned'].apply(lambda x: [cat.strip() for cat in x.split(',')])

# Step 2: Explode the DataFrame so each row is (Instance Name, single GPT category)
topical_exploded_df = topical_df.explode('GPT category split')

# Optional rename for clarity
topical_exploded_df = topical_exploded_df.rename(columns={'GPT category split': 'GPT category single'})

# Step 3: Merge again with rule_topic_data on Instance Name
combined_data_exploded = pd.merge(rule_topic_data, topical_exploded_df, on='Instance Name', how='inner')

from collections import defaultdict

# Step 4: Group: {GPT topic -> instances}
gpt_category_to_instances = defaultdict(list)

for idx, row in combined_data_exploded.iterrows():
    gpt_cat = row['GPT category single']
    instance = row['Instance Name']
    gpt_category_to_instances[gpt_cat].append(instance)

# Step 5: Group: {GPT topic -> rule topic set}
gpt_category_to_rule_topics = {}

for gpt_cat, instances in gpt_category_to_instances.items():
    rule_topics = set()
    
    for instance in instances:
        rt_rows = rule_topic_data[rule_topic_data['Instance Name'] == instance]
        for rt in rt_rows['GPT category set']:
            rule_topics.add(rt)
    
    gpt_category_to_rule_topics[gpt_cat] = rule_topics

summary_df = pd.DataFrame([
    {
        'GPT category': cat,
        'Num Instances': len(set(gpt_category_to_instances[cat])),
        'Rule topics': list(rule_topics)
    }
    for cat, rule_topics in gpt_category_to_rule_topics.items()
])

print(summary_df.head())


# Count of Rule Topics per GPT Topic
rule_topic_counts_per_gpt = (
    combined_data_exploded
    .groupby(['GPT category single', 'GPT category set'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['GPT category single', 'Count'], ascending=[True, False])
)


# --- STEP 1: Clean + Split Rule topic into lists ---
def extract_rule_topics(value):
    if pd.isna(value):
        return []
    clean = re.sub(r'[\[\]\'"]', '', value)  # remove brackets and quotes
    return [item.strip() for item in clean.split(',') if item.strip()]

combined_data_exploded['Rule topic list'] = combined_data_exploded['GPT category set'].apply(extract_rule_topics)

# --- STEP 2: Explode the DataFrame on Rule topic list ---
final_df = combined_data_exploded.explode('Rule topic list').rename(columns={'Rule topic list': 'Rule topic single'})

# Count number of unique instances per GPT category
instances_per_gpt_category = final_df[['GPT category single', 'Instance Name']].drop_duplicates()
instance_counts = instances_per_gpt_category['GPT category single'].value_counts().to_dict()


# --- STEP 3: Group and Count ---
rule_topic_counts_per_gpt = (
    final_df
    .groupby(['GPT category single', 'Rule topic single'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['GPT category single', 'Count'], ascending=[True, False])
)

print(rule_topic_counts_per_gpt.head())

# Clean 'Rule topic single': remove curly braces, quotes, commas, and strip
def clean_topic(text):
    if pd.isna(text):
        return ''
    # Remove unwanted characters
    return re.sub(r"[{}\'\"“”]", '', text).replace(',', '').strip()

final_df['Rule topic single cleaned'] = final_df['Rule topic single'].apply(clean_topic)

# Re-count after cleaning
cleaned_rule_topic_counts_per_gpt = (
    final_df
    .groupby(['GPT category single', 'Rule topic single cleaned'])
    .size()
    .reset_index(name='Count')
    .sort_values(by=['GPT category single', 'Count'], ascending=[True, False])
)

print(cleaned_rule_topic_counts_per_gpt.head())

# Normalize by number of instances in each GPT category
cleaned_rule_topic_counts_per_gpt['Normalized Count'] = cleaned_rule_topic_counts_per_gpt.apply(
    lambda row: row['Count'] / instance_counts[row['GPT category single']],
    axis=1
)

min_val = cleaned_rule_topic_counts_per_gpt['Normalized Count'].min()
max_val = cleaned_rule_topic_counts_per_gpt['Normalized Count'].max()

cleaned_rule_topic_counts_per_gpt['Normalized 0-1'] = (
    (cleaned_rule_topic_counts_per_gpt['Normalized Count'] - min_val) / (max_val - min_val)
)



# Build bipartite graph
B = nx.Graph()

# Add GPT category nodes
gpt_nodes = cleaned_rule_topic_counts_per_gpt['GPT category single'].unique()

B.add_nodes_from(gpt_nodes, bipartite='gpt')

# Add Rule topic nodes
rule_nodes = cleaned_rule_topic_counts_per_gpt['Rule topic single cleaned'].unique()
B.add_nodes_from(rule_nodes, bipartite='rule')

# Add edges with weights
for _, row in cleaned_rule_topic_counts_per_gpt.iterrows():
    B.add_edge(
        row['GPT category single'],
        row['Rule topic single cleaned'],
        weight=row['Count']  # or row['Normalized Count']
    )


# Project onto GPT categories
G_gpt = bipartite.weighted_projected_graph(B, gpt_nodes)

# Add GPT vs Rule attributes
for node in gpt_nodes:
    B.nodes[node]['type'] = 'Instance Description'

for node in rule_nodes:
    B.nodes[node]['type'] = 'Rule Topic'

# Export bipartite graph for Gephi
nx.write_gexf(B, "bipartite_gpt_rule_topics.gexf")
 


# -----------------------------
# 1. Prepare Node Sets
# -----------------------------
gpt_nodes = set(cleaned_rule_topic_counts_per_gpt['GPT category single'].unique())
rule_nodes = set(cleaned_rule_topic_counts_per_gpt['Rule topic single cleaned'].unique())

# -----------------------------
# 2. Build Bipartite Graph
# -----------------------------
B = nx.Graph()

# Add nodes with attributes
B.add_nodes_from(gpt_nodes, bipartite=0, type="GPT Category", group =1)
B.add_nodes_from(rule_nodes, bipartite=1, type="Rule Topic", group =2)

# Add edges with both raw + normalized weight if available
for _, row in cleaned_rule_topic_counts_per_gpt.iterrows():
    B.add_edge(
        row['GPT category single'],
        row['Rule topic single cleaned'],
        weight=float(row['Count']),
        norm=float(row['Normalized Count']) if 'Normalized Count' in row else None
    )

print(f"✅ Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")

# -----------------------------
# 3. Projections
# -----------------------------
# GPT Category Projection
G_gpt = bipartite.weighted_projected_graph(B, gpt_nodes)
nx.set_node_attributes(G_gpt, "GPT Category", "type")
nx.set_node_attributes(G_gpt, 1, "group")

# Rule Topic Projection
G_rules = bipartite.weighted_projected_graph(B, rule_nodes)
nx.set_node_attributes(G_rules, "Rule Topic", "type")
nx.set_node_attributes(G_rules, 2, "group")

print(f"✅ GPT projection: {G_gpt.number_of_nodes()} nodes, {G_gpt.number_of_edges()} edges")
print(f"✅ Rule projection: {G_rules.number_of_nodes()} nodes, {G_rules.number_of_edges()} edges")

# -----------------------------
# 4. Add Useful Metrics
# -----------------------------
def add_metrics(G):
    """Compute and store centrality/degree metrics for Gephi visualization."""
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    nx.set_node_attributes(G, deg, "degree")
    nx.set_node_attributes(G, wdeg, "weighted_degree")
    nx.set_node_attributes(G, bc, "betweenness")

# Add metrics to all graphs
for G in [B, G_gpt, G_rules]:
    add_metrics(G)

# -----------------------------
# 5. Export to Gephi
# -----------------------------
nx.write_gexf(B, "bipartite_gpt_rule_topics.gexf")
nx.write_gexf(G_gpt, "gpt_category_projection.gexf")
nx.write_gexf(G_rules, "rule_topic_projection.gexf")

print("🎉 Exported files for Gephi:")
print(" - bipartite_gpt_rule_topics.gexf (two-mode network)")
print(" - gpt_category_projection.gexf (one-mode GPT)")
print(" - rule_topic_projection.gexf (one-mode Rule)")


# Make sure you have B, gpt_nodes, rule_nodes already defined
plt.figure(figsize=(14, 8))

# Bipartite layout: x-axis separation by bipartite sets
pos = nx.bipartite_layout(B, gpt_nodes)

# Draw GPT categories (left side, blue)
nx.draw_networkx_nodes(B, pos, nodelist=gpt_nodes, 
                       node_color="skyblue", node_size=800, alpha=0.8, label="GPT Categories")

# Draw Rule topics (right side, orange)
nx.draw_networkx_nodes(B, pos, nodelist=rule_nodes, 
                       node_color="orange", node_size=800, alpha=0.8, label="Rule Topics")

# Draw edges (thicker if higher weight)
nx.draw_networkx_edges(B, pos, width=[d["weight"]*0.005 for _,_,d in B.edges(data=True)], alpha=0.4)

# Labels
nx.draw_networkx_labels(B, pos, font_size=8)

plt.title("Bipartite Graph: GPT Categories (left) ↔ Rule Topics (right)")
plt.axis("off")
plt.legend(scatterpoints=1)
plt.show()