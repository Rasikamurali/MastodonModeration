# llm_categorization_accuracy.py
#
# Evaluates the accuracy of GPT-4o-mini rule categorization against human
# annotations from the UCB dataset. Computes three metrics per rule:
# Jaccard similarity, F1 score, and a coverage-based accuracy measure.
# Also computes micro-averaged F1 and exact-match accuracy using sklearn.
#
# Input:  sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv
#         (GPT categories alongside human 'Consolidated_Categories' column)
# Output: printed metric summaries (no file output)

import pandas as pd
import numpy as np
import regex as re
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score

# Load the dataset with both GPT predictions and human ground truth labels
df1 = pd.read_csv(r'sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv')
# df2 = pd.read_csv(r'ucb_mastodon_compiled_categories_sample_1023(2).csv')

print(len(df1))

GPT_category1 = [] 
GPT_category2 = [] 

for row in df1['GPT category']: 
    GPT_category1.append(re.sub(r'A:\s*|[\[\]\'"]', '', row).strip())


df1['GPT category'] = GPT_category1
print(len(df1))

# Step 1: Convert strings to sets of categories
df1['Consolidated_Categories'] = df1['Consolidated_Categories'].apply(
    lambda x: set(x.split(', ')) if isinstance(x, str) else set()
)

df1['GPT category'] = df1['GPT category'].apply(lambda x: set(x.split(', ')) if x else set())

# # Step 1: Convert strings to sets of categories
# df2['Categories'] = df2['Categories'].apply(lambda x: set(x.split(', ')) if x else set())
# df2['GPT category'] = df2['GPT category'].apply(lambda x: set(x.split(', ')) if x else set())

# Step 2: Calculate Jaccard Similarity
def jaccard_similarity(row):
    intersection = row['Consolidated_Categories'].intersection(row['GPT category'])
    union = row['Consolidated_Categories'].union(row['GPT category'])
    return len(intersection) / len(union) if len(union) > 0 else 0

df1['Jaccard Similarity'] = df1.apply(jaccard_similarity, axis=1)
# df2['Jaccard Similarity'] = df2.apply(jaccard_similarity, axis=1)

# Step 3: Calculate F1 Score
def f1_score(row):
    true_labels = row['Consolidated_Categories']
    pred_labels = row['GPT category']
    
    # Calculate precision and recall
    tp = len(true_labels.intersection(pred_labels))  # True Positives
    fp = len(pred_labels - true_labels)  # False Positives
    fn = len(true_labels - pred_labels)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 Score
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

df1['F1 Score'] = df1.apply(f1_score, axis=1)
# df2['F1 Score'] = df2.apply(f1_score, axis=1)

# Step 4: Calculate Accuracy
def accuracy(row):
    true_labels = row['Consolidated_Categories']
    pred_labels = row['GPT category']
    
    # Calculate number of correct predictions
    tp = len(true_labels.intersection(pred_labels))
    total_labels = len(true_labels)
    
    return tp / total_labels if total_labels > 0 else 0

df1['Accuracy'] = df1.apply(accuracy, axis=1)
# df2['Accuracy'] = df2.apply(accuracy, axis=1)


# Display the results
print(df1[['Consolidated_Categories', 'GPT category', 'Jaccard Similarity', 'F1 Score', 'Accuracy']])
print(df1['Jaccard Similarity'].mean())
print(df1['F1 Score'].mean())
print(df1['Accuracy'].mean())



def calculate_metrics_alternative(ground_truth, predictions):
    """
    Calculate F1-score and Accuracy for sets of rule topics manually (alternative approach).

    Args:
    - ground_truth: List of sets, where each set contains the ground truth topics for an instance.
    - predictions: List of sets, where each set contains the predicted topics for an instance.

    Returns:
    - f1: F1-score (manually calculated as micro-averaged across all instances).
    - accuracy: Accuracy across all instances.
    """
    # Ensure ground_truth and predictions have the same length
    assert len(ground_truth) == len(predictions), "Mismatched lengths for ground truth and predictions."
    
    # Convert sets to binary indicator format for multi-label classification
    unique_labels = set().union(*ground_truth, *predictions)  # Get all unique labels
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    def to_binary_matrix(data, label_to_index):
        binary_matrix = np.zeros((len(data), len(label_to_index)), dtype=int)
        for i, labels in enumerate(data):
            for label in labels:
                if label in label_to_index:
                    binary_matrix[i, label_to_index[label]] = 1
        return binary_matrix

    ground_truth_matrix = to_binary_matrix(ground_truth, label_to_index)
    predictions_matrix = to_binary_matrix(predictions, label_to_index)

    # Flatten matrices for micro-averaging
    ground_truth_flat = ground_truth_matrix.flatten()
    predictions_flat = predictions_matrix.flatten()

    # Precision, Recall, and F1 (manually calculated)
    precision = precision_score(ground_truth_flat, predictions_flat, zero_division=0)
    recall = recall_score(ground_truth_flat, predictions_flat, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy
    accuracy = np.mean(np.all(ground_truth_matrix == predictions_matrix, axis=1))

    return f1, accuracy

f1, accuracy = calculate_metrics_alternative(df1['GPT category'], df1['Consolidated_Categories'])
print(f1, accuracy)

