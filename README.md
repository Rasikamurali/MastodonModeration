# Federating Governance: How Community Rules Scale with Mastodon Instances
**Rasika Muralidharan**, Yong-Yeol Ahn, Bao Tran Truong · Indiana University Bloomington, Univeristy of Virginia,  TU Dresden Center Synergy of Systems.

---

## Abstract

This study examines how community size, federation structure, and instance identity shape moderation rule-writing on Mastodon — a federated social network that experienced a large influx of users following Elon Musk's acquisition of Twitter in October 2022. We collected rules from approximately 6,500 instances in October 2023, one year after the Twitter exodus, and applied GPT-4o-mini one-shot prompting to categorize each rule into one of 20 topic types. Using OLS, Negative Binomial, and Beta regression models, we find that larger instances write longer, more topically diverse rule sets, while smaller instances tend toward shorter, more homogeneous rules. We further show that these patterns are stable across robustness checks including rule deduplication, translation validation, alternative activeness measures, and longitudinal Wayback Machine snapshots spanning pre- and post-Elon periods.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
openai
langdetect
textstat
regex
nltk
requests
networkx
python-dotenv
```

Install with:
```bash
pip install -r requirements.txt
```

An OpenAI API key is required for LLM categorization scripts. Set it in a `.env` file:
```
OPEN_AI_KEY=your_key_here
```

All scripts are run from the **project root** directory.

---

## Data

Data files are organized into subfolders under `data/`. See [data/README.md](data/README.md) for full descriptions.

| Subfolder | Contents |
|---|---|
| `data/primary/` | Core datasets: community rules, instance list, LLM category encoding |
| `data/instance meta data/` | Federation peers, weekly activity, instance birth dates |
| `data/llm categorization/` | Raw GPT category outputs and compiled category datasets |
| `data/regression/` | Per-instance regression inputs and outputs |
| `data/deduplicate/` | Deduplicated rule sets and their LLM category encodings |
| `data/topical analysis/` | Instance descriptions, translations, and GPT topic annotations |
| `data/wayback machine/` | Pre- and post-Elon Wayback Machine rule snapshots |

> **Note:** `data/instance meta data/federation_combined.csv` (2.6 GB) and `data/topical analysis/mastodon_topics_wrules.csv` (57 MB) exceed GitHub file size limits and are not tracked in this repository.

---

## Reproducing the Analysis

Scripts should be run in order. All commands are executed from the project root.

### Step 1: Data Collection

```bash
python "code/data collection/full_pipeline.py"                    # Collect rules from Mastodon API
python "code/data collection/federation_data_collection.py"       # Collect federation peer counts
python "code/data collection/instance_birth_info.py"              # Collect instance creation dates
python "code/data collection/weekly_activity_data_collection.py"  # Collect weekly activity metrics
```

### Step 2: Preprocessing

```bash
python code/preprocessing/translate_text.py          # Translate non-English rules to English
python code/preprocessing/getting_annotated_dataset.py  # Prepare annotation samples
```

### Step 3: LLM Rule Categorization

```bash
python code/analysis/main/llm_my_mstdn.py                         # GPT-4o-mini rule categorization
python code/misc/one_shot_encoding_rules.py                        # Binary-encode GPT categories
python code/analysis/main/llm_categorization_accuracy.py          # Evaluate against human annotations
```

### Step 4: Regression Data Preparation

```bash
python code/analysis/main/regression_data_org.py       # Build per-instance regression dataset
python code/analysis/main/regression_wh_topiccounts.py # Add topic count as a DV
```

### Step 5: Main Figures

```bash
python code/analysis/main/figure_1_panel.py                  # Figure 1: Instance size panel
python code/analysis/main/figure_2a_topiccount.py            # Figure 2A: Topic count by size
python code/analysis/main/figure_2b_entropy.py               # Figure 2B: Topic diversity (entropy)
python code/analysis/main/figure_3_heatmap.py                # Figure 3: Topic prevalence heatmap
python code/analysis/main/figure4_word_count_bootstrapped.py # Figure 4: Word count
python code/analysis/main/figure4_rule_count_boostrapped.py  # Figure 4: Rule count
python code/analysis/main/figure4_ttr_bootstrapped.py        # Figure 4: Type-token ratio
python code/analysis/main/figure4_readability_byinstance.py  # Figure 4: Readability
```

### Step 6: Regression Models

```bash
python code/analysis/main/regression_model.py               # OLS models (5 specifications)
python code/analysis/main/negative_binomial_regression.py   # Negative Binomial + Beta regression
```

### Step 7: Robustness Checks

```bash
# Rule deduplication
python code/analysis/robustness/robustness_rules_deduplicate.py
python code/analysis/robustness/robustness_rule_deduplicate_llm_categorization.py
python code/analysis/robustness/robustness_rule_deduplicate.py

# Engagement-based replication
python code/analysis/robustness/robustness_engagement_as_activeness.py

# Wayback Machine / Twitter exodus
python code/analysis/robustness/waybackmachine_preelon.py
python code/analysis/robustness/waybackmachine_postelon2023.py
python code/analysis/robustness/waybackmachine_postelon2024.py
python code/analysis/robustness/wayback_rule_categories.py
python code/analysis/robustness/wayback_machine_analysis_final.py
python code/analysis/robustness/robustness_twitter_exodus.py
```

---

## Repository Structure

```
code/
  data collection/     API scraping (Mastodon, Wayback Machine)
  preprocessing/       Cleaning, formatting, translation
  analysis/
    main/              Primary figures, regression, and LLM scripts
    robustness/        Robustness and replication checks
  misc/                Utility scripts and one-off analyses
data/                  Data files (see data/README.md)
figures/               Output figures (generated by analysis scripts)
```

See [code/README.md](code/README.md) for a full description of every script.

---

## Contact

Rasika Murali · rasimura@iu.edu · Indiana University Bloomington
