# Code File Reference

Scripts are organized into four folders: `analysis/main/`, `analysis/robustness/`, `data collection/`, `preprocessing/`, and `misc/`. This document describes each file's purpose, data inputs, and data outputs.

---

## `analysis/main/`

Core analysis scripts that produce the figures, regression results, and processed datasets used in the paper.

### Rule Categorization (LLM)

| File | Purpose | Input | Output |
|---|---|---|---|
| `llm_my_mstdn.py` | Runs GPT-4o-mini one-shot rule categorization on Mastodon rules in batches | `data/deduplicate/mastodon_rules_deduplicated.csv` | `data/chunks/sampled_categorized_rules_*_0109(1–5).csv` → compiled into `data/llm categorization/categorized_my_rules_deduplicated_full.csv` |
| `llm_test_mastodon_ucb.py` | Tests LLM categorization against the UCB Mastodon annotated dataset | `data/Consolidated_annotated_ucb_dataset.csv` | `data/sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv` |
| `llm_categorization_accuracy.py` | Evaluates GPT category labels against manual annotations; computes Jaccard similarity, F1, and accuracy | `data/sampled_categorized_rules_mastodon_ucb_1211_gpt4o(2).csv` | Console output (metrics) |

### Regression Analysis

| File | Purpose | Input | Output |
|---|---|---|---|
| `regression_data_org.py` | Builds the per-instance regression dataset by computing lexical features and merging with federation and birth date data | `data/primary/community_rules_data.csv`, `data/instance meta data/instance_births_full.csv`, `data/instance meta data/federation_combined.csv` | `data/regression/regression_data_lexicalfeature_fed_birth.csv` |
| `regression_model.py` | OLS regression: user count, federation number, and instance age vs. word count, rule count, TTR, and readability | `data/regression/regression_data_lexicalfeature_fed_birth.csv` | `model1_summary.txt` – `model5_summary.txt` |
| `regression_wh_topiccounts.py` | Extended OLS regression adding number of unique rule topics as a fifth dependent variable | `data/regression/regression_data_lexicalfeature_fed_birth.csv`, `data/primary/one_shot_llm_category_encoding.csv` | `data/regression/regression_lexicalfeature_def_birth_TopicCounts.csv` |
| `negative_binomial_regression.py` | Negative Binomial regression (word count, rule count, topic count) and Beta regression (TTR); Japanese instances excluded from TTR model | `data/regression/regression_data_lexicalfeature_fed_birth.csv`, `data/primary/one_shot_llm_category_encoding.csv`, `data/topical analysis/instance_topics_translated.csv` | `nb_regression_summary.txt`, `beta_regression_summary.txt` |

### Topical Analysis

| File | Purpose | Input | Output |
|---|---|---|---|
| `topical_analysis_GPT4o.py` | Cross-tabulates GPT instance topics vs. rule topics; builds a normalized heatmap | `data/topical analysis/instance_topics_rules_categorization.csv`, `data/llm categorization/llm_category_analysis_data.csv` | Figure (plt.show only) |

### Figures

| File | Purpose | Input | Output |
|---|---|---|---|
| `figure_1_panel.py` | 2×2 panel: CCDF of instance sizes, federation counts, weekly statuses, and logins vs. instance size | `data/primary/community_rules_data.csv`, `data/instance meta data/federation_combined.csv`, `data/instance meta data/weekly_combined.csv` | `figures/figure_1_panel.png/.pdf` |
| `figure_2a_topiccount.py` | Violin + scatter + bootstrap CI: distinct rule topics per instance by size bin | `data/primary/one_shot_llm_category_encoding.csv` | `figures/violin_plot_topic_count_dist.png/.pdf` |
| `figure_2b_entropy.py` | Violin + scatter + bootstrap CI: Shannon entropy of rule topic diversity by size bin | `data/primary/one_shot_llm_category_encoding.csv` | `figures/entropy_plot.png/.pdf` |
| `figure_3_heatmap.py` | Rule topic prevalence heatmap across instance size bins | `data/primary/one_shot_llm_category_encoding.csv`, `data/primary/community_rules_data.csv` | `figures/heatmap.png/.pdf` |
| `figure4_readability_byinstance.py` | Violin + scatter + bootstrap CI: Flesch-Kincaid readability by instance size bin | `data/primary/community_rules_data.csv` | `figures/readability_byinstance.png/.pdf` |
| `figure4_rule_count_boostrapped.py` | Violin + scatter + bootstrap CI: rule count per instance by size bin | `data/primary/community_rules_data.csv` | `figures/rule_count_bootstrapped.png/.pdf` |
| `figure4_ttr_bootstrapped.py` | Violin + scatter + bootstrap CI: type-token ratio by instance size bin | `data/primary/community_rules_data.csv` | `figures/ttr_bootstrapped.png/.pdf` |
| `figure4_word_count_bootstrapped.py` | Violin + scatter + bootstrap CI: total word count per instance by size bin | `data/primary/community_rules_data.csv` | `figures/word_count_bootstrapped.png/.pdf` |

---

## `analysis/robustness/`

Robustness and replication checks validating the main findings.

### Rule Deduplication

| File | Purpose | Input | Output |
|---|---|---|---|
| `robustness_rules_deduplicate.py` | Deduplicates rules within each instance using Jaccard similarity at the 99th-percentile threshold | `data/primary/community_rules_data.csv` | `data/deduplicate/mastodon_rules_deduplicated.csv` |
| `robustness_rule_deduplicate_llm_categorization.py` | GPT-4o-mini categorization on deduplicated rules to verify deduplication does not shift topic distribution | `data/deduplicate/mastodon_rules_deduplicated.csv` | `data/llm categorization/categorized_my_rules_deduplicated_full.csv` |
| `robustness_rule_deduplicate.py` | Replicates Figure 2A (topic count distribution) using deduplicated rules | `data/deduplicate/deduplicated_oneshot.csv` | `figures/robustness_topic_count_deduplicated_rules.png/.pdf` |

### Instance Topic Analysis (Robustness)

| File | Purpose | Input | Output |
|---|---|---|---|
| `robustness_instance_description_exploration.py` | Merges community rules with instance metadata to build the full rules-with-topics dataset; explores description frequency distribution | `data/primary/complete_instance_list.csv`, `data/primary/community_rules_data.csv` | `data/topical analysis/mastodon_topics_wrules.csv` (currently commented out) |
| `robustness_instance_topic_translate.py` | Translates instance short descriptions into English for GPT topic classification | `data/topical analysis/mastodon_topics_wrules.csv` | `data/topical analysis/instance_topics_translated.csv` |
| `robustness_instance_topic_categorization.py` | GPT-4o-mini classification of instance descriptions into 26 topic types, run in batches | `data/topical analysis/instance_topics_translated.csv`, `data/topical analysis/sampled_instances_topics.csv` | `data/topical analysis/instance_topics_rules_categorization.csv` |
| `robustness_instance_topic_rule_variation.py` | Computes within- vs. across-instance variance of rule topics by GPT instance category | `data/topical analysis/instance_topics_rules_categorization.csv`, `data/llm categorization/llm_category_analysis_data.csv` | Two figures (plt.show only) |

### Engagement as Activeness

| File | Purpose | Input | Output |
|---|---|---|---|
| `robustness_engagement_as_activeness.py` | Replicates lexical feature plots using post volume instead of user count as the size proxy | `data/primary/community_rules_data.csv`, `data/instance meta data/weekly_combined.csv` | `figures/engagement_*.png/.pdf` |

### Wayback Machine / Twitter Exodus

| File | Purpose | Input | Output |
|---|---|---|---|
| `wayback_machine_snapshot_availability.py` | Checks Wayback Machine CDX API for snapshot availability across all instances | `data/primary/community_rules_data.csv` | `data/chunks/snapshot_availability_*.csv` → compiled into `data/wayback machine/snapshot_availability_full.csv` |
| `waybackmachine_preelon.py` | Collects pre-Elon (Sept–Oct 2022) rules from Wayback Machine for the 126 seed instances | `data/wayback machine/non_personal_preelon_notnull.csv` | `data/wayback machine/non_personal_preelon_2022.csv` |
| `waybackmachine_postelon2023.py` | Collects post-Elon (Jan 2023–Jan 2024) rules from Wayback Machine for the 126 seed instances | `data/wayback machine/non_personal_preelon_notnull.csv` | `data/wayback machine/non_personal_postelon_2023(1).csv` |
| `waybackmachine_postelon2024.py` | Collects post-Elon (Jan–Feb 2024) rules from Wayback Machine for the 126 seed instances | `data/wayback machine/non_personal_preelon_notnull.csv` | `data/wayback machine/non_personal_postelon_2024(1).csv` |
| `wayback_rule_categories.py` | GPT-4o-mini categorization of rules from all three time periods | `data/wayback machine/non_personal_preelon_notnull.csv`, `data/wayback machine/non_personal_preelon_2022.csv`, `data/wayback machine/non_personal_postelon_2023(1).csv`, `data/wayback machine/non_personal_postelon_2024(1).csv` | `data/wayback machine/rule_category_comparison.csv` |
| `wayback_machine_analysis_final.py` | Merges pre- and post-Elon snapshots; compares rule and word count changes; GPT category comparison across periods | Wayback CSVs, `data/wayback machine/rule_category_comparison.csv` | `data/wayback machine/instance_rule_changes_comparison.csv`, figures |
| `robustness_twitter_exodus.py` | Statistical analysis of rule/word count changes and user count jumps pre/post Elon using bootstrapped CIs and t-tests | Wayback CSVs | Printed t-test and bootstrap CI results |
| `twitter_exodus_category.py` | Analyzes stability of GPT rule topic categories across Wayback periods using Jaccard similarity | `data/wayback machine/rule_category_comparison.csv` | Printed statistics |
| `wayback_machine_scraping.py` | Earlier multi-period scraping scaffold; fetches rules across 4 time periods for a stratified sample | `data/primary/community_rules_data.csv` | `sampled_instances_norm_history_preelon.csv` |

### Other Robustness

| File | Purpose | Input | Output |
|---|---|---|---|
| `wrongcolor_fig2a.py` | Earlier version of the topic count violin plot with incorrect color scheme; kept for reference | `data/deduplicate/deduplicated_oneshot.csv` | `figures/violin_plot_topic_count_dist.png/.pdf` |

---

## `data collection/`

Scripts that collect raw data from Mastodon APIs and external services.

| File | Purpose | Input | Output |
|---|---|---|---|
| `full_pipeline.py` | End-to-end pipeline: enumerates instances via instances.social, fetches rules, stratifies by user count | Mastodon API (`instances.social`, instance rule endpoints) | `data/primary/community_rules_data.csv`, `data/primary/complete_instance_list.csv` |
| `federation_data_collection.py` | Collects federation peer lists for each instance via `/api/v1/instance/peers` | `data/primary/community_rules_data.csv` | `data/instance meta data/federation_combined.csv` |
| `instance_birth_info.py` | Queries instance creation dates from `contact_account.created_at` via the Mastodon API | `data/primary/translated_rules_dataset.csv` | `data/instance meta data/instance_births_full.csv` |
| `weekly_activity_data_collection.py` | Collects weekly activity metrics (statuses, logins, registrations) via `/api/v1/instance/activity` | `data/primary/community_rules_data.csv` | `data/instance meta data/weekly_combined.csv` |

---

## `preprocessing/`

Scripts that clean and prepare raw data before analysis.

| File | Purpose | Input | Output |
|---|---|---|---|
| `dataorg.py` | Concatenates per-stratum rule CSVs and assigns instance group labels | `data/instance rules/instance_rules_*.csv` | `merged_instance_data.csv` |
| `final_cleaning.py` | Filters error rows, flattens to one-rule-per-row format, and detects language | `data/instance_rules_1_15.csv` | `5001_final_formatted_data.csv` |
| `translate_text.py` | Detects non-English rules and translates them to English using the Google Translate API | `data/final_rules.csv` | `data/primary/translated_rules_dataset.csv` |
| `getting_annotated_dataset.py` | Creates annotation sample sets: 50 shared rules for inter-annotator reliability + 100 rules each for two annotators | `data/primary/translated_rules_dataset.csv` | `Rasika_annotation_full.csv`, `Bao_annotation_full.csv` |

---

## `misc/`

Utility and one-off scripts for data fixes, robustness checks, and statistical tests.

| File | Purpose | Input | Output |
|---|---|---|---|
| `one_shot_encoding_rules.py` | Processes raw GPT output: cleans category strings, binary-encodes all 20 categories as one-hot columns | `data/llm categorization/categorized_my_rules_deduplicated_full.csv` | `data/primary/one_shot_llm_category_encoding.csv` |
| `one_shot_encoding_deduplicate_rules.py` | Same encoding pipeline applied specifically to the deduplicated rule set | `data/deduplicate/deduplicated_oneshot.csv` | `data/primary/one_shot_llm_category_encoding.csv` |
| `sampling_instance_topic_annotation.py` | Draws a random sample of 200 instances for manual topic annotation | `data/topical analysis/instance_topics_translated.csv` | `data/topical analysis/sampled_instances_topics.csv` |
| `birth_analysis.py` | Compares rule topic distributions of pre- vs. post-October 2022 instances | `data/instance meta data/instance_births_full.csv`, `data/llm categorization/llm_category_analysis_data.csv` | Bar plot (plt.show only) |
| `consolidated_mstdn_ucb_dataset.py` | Converts binary annotation columns to comma-separated category strings per rule | `data/llm categorization/MastodonRules-sample-cleaned-annotated_ucboulder.csv` | `data/Consolidated_annotated_ucb_dataset.csv` |
| `counting_instance.py` | Counts instances per log-scale user count bin and merges with LLM category encoding | `data/primary/community_rules_data.csv`, `data/primary/one_shot_llm_category_encoding.csv` | `data/deduplicate/deduplicated_oneshot.csv` |
| `error_log_analysis.py` | Counts empty, error, and HTTP-failure rows in the raw rules file | `data/instance_rules_20_strat_0306.csv` | Printed counts |
| `fixing_regression_data.py` | Early predecessor to `regression_data_org.py`; incomplete data transformation for regression input | `data/primary/community_rules_data.csv`, federation/birth data | Intermediate CSVs |
| `lexical_diversity_nltk.py` | Computes POS-filtered lexical diversity (nouns, verbs, adjectives, pronouns only) per instance | `data/primary/community_rules_data.csv` | `ld_nltk.png`, `ld_nltk.pdf` |
| `robustness_eval.py` | Summarizes and visualizes translation robustness scores | `data/primary/translation_robustness_scores(2).csv` | Console output / figures |
| `robustness_instance_topic_bipartite.py` | Constructs a bipartite network linking instance topics to rule topics; exports GEXF files | `data/topical analysis/instance_topics_rules_categorization.csv`, `data/llm categorization/llm_category_analysis_data.csv` | `bipartite_gpt_rule_topics.gexf`, `gpt_category_projection.gexf`, `rule_topic_projection.gexf` |
| `statistical_test_engvsnoneng.py` | Welch's t-tests comparing English vs. non-English instances on word count, TTR, and rule count | `data/primary/community_rules_data.csv` | Printed test statistics |
| `translated_robustness_check.py` | Validates translation quality by comparing Flesch-Kincaid readability on original vs. translated rules | `data/primary/community_rules_data.csv` | `data/primary/translation_robustness_scores(2).csv` |
