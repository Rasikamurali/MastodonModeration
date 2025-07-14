*IN process of adding the new the data*


`raw`: raw data. read-only. 
<li>complete_instance_list.csv: All instances (ID, Name, description, User Count)</li>
<li>federation_combined.csv: All instances (federation number, federation instances) </li>
<li>instance_births_full.csv: Instance Names and date of birth (only for instances that had rules)</li>

<br>

`derived`: datasets derived from the raw data. 
<li>community_rules_data.csv: Rules of each instance of Mastodon with User count and rule count </li>
<li>categorized_my_rules_gpt4omini_full.csv: LLM categories for each rule</li>
<li>Entroppy check.csv: Shannon Entropy score for each set of rules of an instance</li>
<li>regression_data_lexicalfeature_fed_birth.csv: Instance Name, lexical features (and scores) </li>

<br>

`additional`: misc. additional datasets. 
<li>translation_robustness_scores.csv: Flesch Kincaid score of rules in non-Enlgish languages</li>
<li>MastodonRules-sample-cleaned-annotated.csv: Annotated set of rules</li>
