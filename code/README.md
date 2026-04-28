# Code File Reference

Scripts are organized into four folders: `analysis/main/`, `analysis/robustness/`, `data collection/`, `preprocessing/`, and `misc/`. This document describes each file's purpose, data inputs, and data outputs.

---

## `analysis/main/`

Core analysis scripts that produce the figures, regression results, and processed datasets used in the paper.

### Rule Categorization (LLM)

| File | Purpose | Input | Output |
|---|---|---|---|
| `llm_my_mstdn.py` | Runs GPT-4o-mini one-shot rule categorization on Mastodon rules in batches | `data/mastodon_rules_deduplicated.csv` | `data/chunks/sampled_categorized_rules_*_0109(1–5).csv` → compiled into `data/categorized_my_rules_gpt4omini_full.csv` |
| `llm_test_mastodon_ucb.py` | Tests LLM categorization against the UCB Mastodon annotated dataset | `data/Consolidated_annotated_ucb_dataset.csv` | `data/sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv` |
| `llm_categorization_accuracy.py` | Evaluates GPT category labels against manual annotations; computes Jaccard similarity, F1, and accuracy | `data/sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv` | Console output (metrics) |

### Regression Analysis

| File | Purpose | Input | Output |
|---|---|---|---|
| `regression_data_org.py` | Builds the per-instance regression dataset by computing lexical features and merging with federation and birth date data | `data/community_rules_data.csv`, `data/instance_births_full.csv`, `data/federation_data_combined.csv` | `data/regression_data_lexicalfeature_fed_birth.csv` |
| `regression_model.py` | OLS regression: user count, federation number, and instance age vs. word count, rule count, TTR, and readability | `data/regression_data_lexicalfeature_fed_birth.csv` | `model1_summary.txt` – `model5_summary.txt` |
| `regression_wh_topiccounts.py` | Extended OLS regression adding number of unique rule topics as a fifth dependent variable | `data/regression_data_lexicalfeature_fed_birth.csv`, `data/one_shot_llm_category_encoding.csv` | `model1_summary.txt` – `model5_summary.txt` |
| `negative_binomial_regression.py` | Negative Binomial regression (word count, rule count, topic count) and Beta regression (TTR); Japanese instances excluded from TTR model | `data/regression_data_lexicalfeature_fed_birth.csv`, `data/one_shot_llm_category_encoding.csv`, `data/instance_topics_translated.csv` | `nb_regression_summary.txt`, `beta_regression_summary.txt` |

### Topical Analysis

| File | Purpose | Input | Output |
|---|---|---|---|
| `sampling_instance_topic_annotation.py` | Random sample of 200 instances for manual topic annotation | `data/instance_topics_translated.csv` | `data/sampled_instances_topics.csv` |
| `topical_analysis_GPT4o.py` | Cross-tabulates GPT instance topics vs. rule topics; builds a normalized heatmap | `data/sampled_annotations_GPT_category(4)_full.csv`, `data/llm_category_analysis_data.csv` | Figure (plt.show only) |

### Figures

| File | Purpose | Input | Output |
|---|---|---|---|
| `figure_1_panel.py` | 2×2 panel: CCDF of instance sizes, federation counts, weekly statuses, and logins vs. instance size | `data/community_rules_data.csv`, `data/federation_data_combined.csv`, `data/weekly_combined.csv` | `figures/figure_1_panel.png/.pdf` |
| `figure_2a_topiccount.py` | Violin + scatter + bootstrap CI: distinct rule topics per instance by size bin | `data/one_shot_llm_category_encoding.csv` | `figures/violin_plot_topic_count_dist.png/.pdf` |
| `figure_2b_entropy.py` | Violin + scatter + bootstrap CI: Shannon entropy of rule topic diversity by size bin | `data/one_shot_llm_category_encoding.csv` | `figures/entropy_plot.png/.pdf` |
| `figure_3_heatmap.py` | Rule topic prevalence heatmap across instance size bins | `data/one_shot_llm_category_encoding.csv`, `data/community_rules_data.csv` | `figures/heatmap.png/.pdf` |
| `figure4_readability_byinstance.py` | Violin + scatter + bootstrap CI: Flesch-Kincaid readability by instance size bin | `data/community_rules_data.csv` | `figures/readability_byinstance.png/.pdf` |
| `figure4_rule_count_boostrapped.py` | Violin + scatter + bootstrap CI: rule count per instance by size bin | `data/community_rules_data.csv` | `figures/rule_count_bootstrapped.png/.pdf` |
| `figure4_ttr_bootstrapped.py` | Violin + scatter + bootstrap CI: type-token ratio by instance size bin | `data/community_rules_data.csv` | `figures/ttr_bootstrapped.png/.pdf` |
| `figure4_word_count_bootstrapped.py` | Violin + scatter + bootstrap CI: total word count per instance by size bin (instances with word count < 100 only) | `data/community_rules_data.csv` | `figures/word_count_bootstrapped.png/.pdf` |

---

## `analysis/robustness/`

Robustness and replication checks validating the main findings.

### Rule Deduplication

| File | Purpose | Input | Output |
|---|---|---|---|
| `robustness_rules_deduplicate.py` | Deduplicates rules within each instance using Jaccard similarity at the 99th-percentile threshold | `data/community_rules_data.csv` | `mastodon_rules_deduplicated.csv` |
| `robustness_rule_deduplicate_llm_categorization.py` | GPT-4o-mini categorization on deduplicated rules to verify deduplication does not shift topic distribution | `data/mastodon_rules_deduplicated.csv` | `data/categorized_my_rules_gpt4omini_full.csv` |
| `robustness_rule_deduplicate.py` | Replicates Figure 2A (topic count distribution) using deduplicated rules | `data/deduplicated_oneshot.csv` | `figures/robustness_topic_count_deduplicated_rules.png/.pdf` |

### Instance Topic Analysis (Robustness)

| File | Purpose | Input | Output |
|---|---|---|---|
| `robustness_instance_topic_translate.py` | Translates instance short descriptions into English for GPT topic classification | `data/mstdn_topics_wnorms.csv` | `data/instance_topics_translated.csv` |
| `robustness_instance_topic_categorization.py` | GPT-4o-mini classification of instance descriptions into 26 topic types, run in batches | `data/instance_topics_translated.csv`, `data/sampled_instances_topics.csv` | `data/sampled_annotations_GPT_category(4)_5000_6000.csv` |
| `robustness_instance_topic_rule_variation.py` | Computes within- vs. across-instance variance of rule topics by GPT instance category | `data/sampled_annotations_GPT_category(4)_full.csv`, `data/llm_category_analysis_data.csv` | Two figures (plt.show only) |
| `robustness_instance_description_exploration.py` | Explores frequency distribution of translated instance descriptions | `data/complete_instance_list.csv`, `data/community_rules_data.csv`, `data/instance_topics_translated.csv` | Printed counts |

### Engagement as Activeness

| File | Purpose | Input | Output |
|---|---|---|---|
| `robustness_engagement_as_activeness.py` | Replicates lexical feature plots (word count, TTR, rule count, readability) using post volume instead of user count as the size proxy | `data/community_rules_data.csv`, `data/weekly_combined.csv` | `figures/engagement_word_count.png/.pdf`, `figures/engagement_ttr.png/.pdf`, `figures/engagement_rule_count.png/.pdf`, `figures/engagement_readability.png/.pdf` |

### Wayback Machine / Twitter Exodus

| File | Purpose | Input | Output |
|---|---|---|---|
| `wayback_machine_scraping.py` | Scrapes Wayback Machine CDX API to find available snapshots for all instances across 4 time periods | `data/community_rules_data.csv` | `data/chunks/snapshot_availability_*.csv` → compiled into `data/snapshot_availability_full.csv` |
| `waybackmachine_preelon.py` | Collects post-Elon (Jan–Feb 2024) rules from Wayback Machine for the 126 sampled instances | `non_personal_preelon_notnull.csv` | `non_personal_postelon_2024(1).csv` |
| `wayback_rule_categories.py` | GPT-4o-mini categorization of rules from 3 time periods (pre-Elon, post-Elon 2023, post-Elon 2024) | 4 Wayback CSVs | `data/chunks/rule_category_comparison.csv` |
| `wayback_machine_analysis_final.py` | Merges pre- and post-Elon rule snapshots; compares rule and word count changes; GPT category comparison across periods | 4 Wayback CSVs, `data/chunks/rule_category_comparison.csv` | `instance_rule_changes_comparison.csv`, figures |
| `robustness_twitter_exodus.py` | Statistical analysis of rule and word count changes and user count jumps pre/post Elon using bootstrapped CIs and t-tests | 4 Wayback CSVs | Printed t-test and bootstrap CI results |
| `twitter_exodus_category.py` | Analyzes stability of GPT rule topic categories across Wayback periods using Jaccard similarity | `data/chunks/rule_category_comparison.csv` | Printed statistics |

### Other Robustness

| File | Purpose | Input | Output |
|---|---|---|---|
| `wrongcolor_fig2a.py` | Earlier version of the topic count violin plot with incorrect color scheme; kept for reference | `data/deduplicated_oneshot.csv` | `figures/violin_plot_topic_count_dist.png/.pdf` |

---

## `data collection/`

Scripts that collect raw data from Mastodon APIs and external services.

| File | Purpose | Input | Output |
|---|---|---|---|
| `full_pipeline.py` | End-to-end pipeline: enumerates instances via instances.social, fetches rules, stratifies by user count | Mastodon API (`instances.social`, instance rule endpoints) | `data/community_rules_data.csv`, `data/complete_instance_list.csv` |
| `federation_data_collection.py` | Collects federation peer lists for each instance via `/api/v1/instance/peers` | `data/complete_strat20_translated_rules.csv` | `data/federation_data_strat20.csv` → compiled into `data/federation_data_combined.csv` |
| `instance_birth_info.py` | Queries instance creation dates from `contact_account.created_at` via the Mastodon API | `data/translated_rules_dataset.csv`, `data/complete_strat20_translated_rules.csv` | `data/instance_births_full.csv` |
| `weekly_activity_data_collection.py` | Collects weekly activity metrics (statuses, logins, registrations) via `/api/v1/instance/activity` | `data/complete_strat20_translated_rules.csv` | `data/weekly_combined.csv` |

---

## `preprocessing/`

Scripts that clean and prepare raw data before analysis.

| File | Purpose | Input | Output |
|---|---|---|---|
| `dataorg.py` | Concatenates 7 per-stratum rule CSVs and assigns instance group labels | `data/instance rules/instance_rules_*.csv` | `merged_instance_data.csv` |
| `final_cleaning.py` | Filters error rows, flattens to one-rule-per-row format, and detects language | `data/instance_rules_1_15.csv` | `5001_final_formatted_data.csv`, `Tester.csv`, `checker.csv` |
| `translate_text.py` | Detects non-English rules and translates them to English using the Google Translate API | `data/final_rules.csv` | `data/translated_final2.csv` |
| `getting_annotated_dataset.py` | Creates annotation sample sets: 50 shared rules for inter-annotator reliability + 100 rules each for two annotators | `data/translated_rules_dataset.csv` | `Rasika_annotation_full.csv`, `Bao_annotation_full.csv` |

---

## `misc/`

Utility and one-off scripts for data fixes, robustness checks, and statistical tests.

| File | Purpose | Input | Output |
|---|---|---|---|
| `analysis_random.py` | Processes raw GPT output, applies category normalization mapping, and binary-encodes category labels | `sampled_categorized_rules_my_mastodon_gpt4omini_0326_wUserCount.csv` | `temp2.csv` |
| `birth_analysis.py` | Compares rule topic distributions of pre- vs. post-October 2022 instances | `data/instance_births_full.csv`, `data/llm_category_analysis_data.csv` | Bar plot (plt.show only) |
| `consolidated_mstdn_ucb_dataset.py` | Converts binary annotation columns to comma-separated category strings per rule | `data/MastodonRules-sample-cleaned-annotated.csv` | `data/Consolidated_annotated_ucb_dataset.csv` |
| `counting_instance.py` | Counts instances per log-scale user count bin and merges with LLM category encoding | `data/community_rules_data.csv`, `data/one_shot_llm_category_encoding.csv` | `data/deduplicated_oneshot.csv` |
| `error_log_analysis.py` | Counts empty, error, and HTTP-failure rows in the raw rules file | `data/instance_rules_20_strat_0306.csv` | Printed counts |
| `fixing_regression_data.py` | Early predecessor to `regression_data_org.py`; incomplete data transformation for regression input | `data/community_rules_data.csv`, federation/birth data, category files | `regression_wTopicCounts.csv` |
| `lame.py` | Diagnostic script; earlier version of the `counting_instance.py` logic | `data/community_rules_data.csv`, `data/one_shot_llm_category_encoding.csv` | `data/deduplicated_oneshot.csv` |
| `lexical_diversity_nltk.py` | Computes POS-filtered lexical diversity (nouns, verbs, adjectives, pronouns only) per instance | `data/community_rules_data.csv` | `ld_nltk.png`, `ld_nltk.pdf` |
| `robustness_eval.py` | Summarizes and visualizes translation robustness scores | `data/translation_robustness_scores(2).csv` | Console output / figures |
| `robustness_instance_topic_bipartite.py` | Constructs a bipartite network linking instance topics to rule topics | `data/sampled_annotations_GPT_category(4)_full.csv`, `data/llm_category_analysis_data.csv` | `.gexf` network files |
| `statistical_test_engvsnoneng.py` | Welch's t-tests comparing English vs. non-English instances on word count, TTR, and rule count | `data/community_rules_data.csv` | Printed test statistics |
| `translated_robustness_check.py` | Validates translation quality by comparing Flesch-Kincaid readability on original vs. translated rules | `data/community_rules_data.csv` | Printed statistics |
