# robustness_rules_deduplicate.py
#
# Deduplicates rules within each Mastodon instance using Jaccard token
# similarity. The similarity threshold is set at the 99th percentile of
# a baseline distribution of random rule-pair similarities across the
# full dataset. Ensures that nearly-identical rules do not inflate topic
# count or lexical diversity metrics.
#
# Input:  data/community_rules_data.csv       (main rules dataset with 'translated text')
# Output: mastodon_rules_deduplicated.csv     (one row per unique rule per instance)

import re
import random
import numpy as np
from typing import Dict, List
import pandas as pd


def tokenize(rule: str) -> set:
    """
    Convert a rule string into a set of tokens for Jaccard similarity.
    """
    #make sure rule is str 
    rule = str(rule)
    rule = rule.lower()
    return set(re.findall(r"\w+", rule))


def jaccard(a: set, b: set) -> float:
    """
    Compute Jaccard similarity between two sets.
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def collect_all_rules(instances: Dict[str, List[str]]) -> List[str]:
    """
    Flatten all instance rule sets into one list.
    """
    all_rules = []
    for rules in instances.values():
        all_rules.extend(rules)
    return all_rules


def sample_random_jaccards(
    rule_sets: List[set],
    n_samples: int = 10000,
    seed: int = 42
) -> np.ndarray:
    """
    Sample Jaccard similarities between random pairs of rules.
    """
    random.seed(seed)
    n = len(rule_sets)
    sims = []

    if n < 2:
        return np.array([])

    for _ in range(n_samples):
        i, j = random.sample(range(n), 2)
        sims.append(jaccard(rule_sets[i], rule_sets[j]))

    return np.array(sims)


def choose_threshold(
    baseline_sims: np.ndarray,
    percentile: float = 99
) -> float:
    """
    Choose similarity threshold from baseline distribution.
    """
    if len(baseline_sims) == 0:
        return 1.0
    return float(np.percentile(baseline_sims, percentile))


def deduplicate_instance_rules(
    rules: List[str],
    threshold: float
) -> List[str]:
    """
    Deduplicate rules within a single instance based on Jaccard similarity.
    """
    tokenized = [tokenize(r) for r in rules]
    kept_indices = []

    for i, tokens_i in enumerate(tokenized):
        duplicate = False
        for j in kept_indices:
            if jaccard(tokens_i, tokenized[j]) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept_indices.append(i)

    return [rules[i] for i in kept_indices]


def deduplicate_all_instances_and_save(
    df: pd.DataFrame,
    instance_col: str = "Instance Name",
    rule_col: str = "translated text",
    output_path: str = "deduplicated_rules.csv",
    n_baseline_samples: int = 10000,
    baseline_percentile: float = 99
):
    """
    Deduplicate rules within each instance using a global similarity baseline
    and save the resulting DataFrame.
    """

    # ---- build instances dict from DataFrame ----
    instances = (
        df.groupby(instance_col)[rule_col]
          .apply(list)
          .to_dict()
    )

    # ---- Collect and tokenize all rules (UNCHANGED) ----
    all_rules = collect_all_rules(instances)
    all_rule_sets = [tokenize(r) for r in all_rules]

    # ---- Compute baseline similarities (UNCHANGED) ----
    baseline_sims = sample_random_jaccards(
        all_rule_sets,
        n_samples=n_baseline_samples
    )

    # ---- Choose threshold (UNCHANGED) ----
    threshold = choose_threshold(
        baseline_sims,
        percentile=baseline_percentile
    )

    print("Baseline statistics:")
    print(f"  mean similarity: {baseline_sims.mean():.3f}")
    print(f"  95th percentile: {np.percentile(baseline_sims, 95):.3f}")
    print(f"  99th percentile: {np.percentile(baseline_sims, 99):.3f}")
    print(f"  chosen threshold: {threshold:.3f}")

    # ---- count instances with highly similar rules ----
    instances_with_duplicates = 0

    for rules in instances.values():
        tokenized = [tokenize(r) for r in rules]
        found_duplicate = False

        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                if jaccard(tokenized[i], tokenized[j]) >= threshold:
                    found_duplicate = True
                    break
            if found_duplicate:
                break

        if found_duplicate:
            instances_with_duplicates += 1


    # ---- Deduplicate per instance (UNCHANGED LOGIC) ----
    rows = []
    stats = {}

    for instance, rules in instances.items():
        deduped = deduplicate_instance_rules(rules, threshold)

        stats[instance] = {
            "before": len(rules),
            "after": len(deduped),
            "removed": len(rules) - len(deduped)
        }

        # NEW: rebuild DataFrame rows
        for rule in deduped:
            rows.append({
                instance_col: instance,
                rule_col: rule
            })

    # ---- Create and save DataFrame (NEW) ----
    df_clean = pd.DataFrame(rows)
    df_clean.to_csv(output_path, index=False)

    print(f"\nSaved deduplicated DataFrame to: {output_path}")
    print(f"Total rules before: {len(df)}")
    print(f"Total rules after:  {len(df_clean)}")

    return df_clean, stats, threshold


df = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\community_rules_data.csv')
print(df.columns)

instances = {}
for _, row in df.iterrows():
    instance = row['Instance Name']
    if instance not in instances:
        instances[instance] = []
    instances[instance].append(row['translated text'])


df_clean, stats, threshold = deduplicate_all_instances_and_save(
    df,
    output_path="mastodon_rules_deduplicated.csv"
)


