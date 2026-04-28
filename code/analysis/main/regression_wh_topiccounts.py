# regression_wh_topiccounts.py
#
# Extends regression_model.py by adding Number of Unique Rule Topics as a
# fifth dependent variable. Runs the same five OLS model specifications
# (user count, federation, age, and combinations) across all five DVs.
#
# Input:  data/one_shot_llm_category_encoding.csv        (GPT category encodings)
#         data/regression_data_lexicalfeature_fed_birth.csv  (lexical + federation + birth)
# Output: model1_summary.txt ... model5_summary.txt  (OLS regression summaries)

#Import required libraries
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import ast
from itertools import chain
from scipy.stats import kruskal, spearmanr
plt.style.use(r'C:\Users\rasik\Documents\Independent Study\code\main_stylesheet.mplstyle')
import ast
from scipy.stats import zscore
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import statsmodels.api as sm
from collections import Counter
from datetime import datetime, timezone


# Color-blind friendly palette orange, purple, blue, teal, yellow, pink
PALETTE = ["#F18447", "#550F6B",  "#3863AC", "#209B8A", "#F8D625", "#BC3684"]

# Load binary-encoded LLM category data; derive number of unique topics per instance
updated_df = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\one_shot_llm_category_encoding.csv')

# Ensure 'GPT category set' contains a set of words
def ensure_set_of_words(value):
    try:
        # Safely evaluate the string to convert it into a Python object
        evaluated_value = ast.literal_eval(value) if isinstance(value, str) else value
        # Convert to set if it's not already
        if isinstance(evaluated_value, (list, set)):
            return set(evaluated_value)
        else:
            # If the value is not a list or set, return an empty set
            return set()
    except (ValueError, SyntaxError):
        # If there's an issue with literal_eval, return an empty set
        return set()

# Apply the function to the 'GPT category set' column
updated_df['GPT category set'] = updated_df['GPT category set'].apply(ensure_set_of_words)


#Calculate number of topics per instance
topics_per_instance = updated_df.groupby('Instance Name').agg(
    Topics=('GPT category set', lambda x: {cat: sum(cat in s for s in x) for s in x for cat in s}),
    User_Count=('User Count', 'first')
).reset_index()

#Adding log bins based on user count 
bins = [1, 10, 100, 1000, 10000, 100000, 1000000]  
bin_labels = [r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"]


# Assign each instance to a user count bin
topics_per_instance['User Count Bin'] = pd.cut(topics_per_instance['User_Count'], bins=bins, labels=bin_labels, right=False)


#Adding new column to indicate number of unique topics per instance 
topics_per_instance['Number of Unique Topics'] = topics_per_instance['Topics'].apply(len)

#print(topics_per_instance)

df = pd.read_csv(r'C:\Users\rasik\Documents\Independent Study\data\regression_data_lexicalfeature_fed_birth.csv')
print(len(topics_per_instance), len(df))

print(topics_per_instance.columns)
print(df.columns)

#merge dataframes
merged_df = pd.merge(df, topics_per_instance[['Instance Name', 'Number of Unique Topics']], on='Instance Name', how='inner')
print(len(merged_df))

#Regression analysis 
# Model 1: User count versus features 
# Model 2: Federation count versus features
# Model 3: Interactions terms versus features 

merged_df = merged_df.rename(columns={'User Count_x': 'User Count'})

print(merged_df.columns)
merged_df['Log_User_Count'] = np.log10(merged_df['User Count'])
print(len(merged_df))


merged_df = merged_df[merged_df['federating number'] > 0]  # Keep only positive values
merged_df['Log_fed_num'] = np.log10(merged_df['federating number'])
print(len(merged_df))

births = (Counter(merged_df['birth']))
# print(births.most_common(30))   

merged_df = merged_df[~merged_df['birth'].str.contains("Failed to retrieve instance information. Status code: ", na=False)]

# Replace 'Unknown error' and 'Data not available' with NaN
merged_df['birth'] = merged_df['birth'].replace('Unknown error', np.nan)
merged_df['birth'] = merged_df['birth'].replace('Data not available', np.nan)
# Convert 'birth' column to datetime (coercing errors to NaT)
merged_df['birth'] = pd.to_datetime(merged_df['birth'], errors='coerce', utc=True)

# # Drop rows where 'birth' is NaT (invalid datetime values)
merged_df = merged_df.dropna(subset=['birth'])
print(len(merged_df))
print(merged_df.columns)
merged_df = merged_df.dropna(subset=['word count', 'rule count', 'TTR', 'Fk Score', 'Number of Unique Topics'])




# # Define predictors (only log-transformed User Count)
X_User = merged_df[['Log_User_Count']]
X_User = sm.add_constant(X_User)  # Add intercept

# Define multiple dependent variables (lexical features)
Y = merged_df[['word count', 'rule count', 'TTR', 'Fk Score', 'Number of Unique Topics']]

#Model 1: User Count versus Lexical Features

# # Fit regression models separately for each lexical feature
model1 = {col: sm.OLS(Y[col], X_User).fit() for col in Y.columns}

with open("model1_summary.txt", "w") as f:  # overwrite only once, at the beginning
    f.write("Model 1: User Count vs Lexical Features\n")

for feature, model in model1.items():
    print(f"\nRegression Results for {feature}:\n")
    print(model.summary())
    with open("model1_summary.txt", "a") as f:  # append here
        f.write(f"\nRegression Results for {feature}:\n")
        f.write(model.summary().as_text())
        f.write("\n" + "-"*80 + "\n")  # separator


#Model 2: Federating Number versus Lexical Features 

# Define predictors (only log-transformed User Count)
X_fed = merged_df[['Log_fed_num']]
X_fed = sm.add_constant(X_fed)  # Add intercept

# Fit regression models separately for each lexical feature
model2 = {col: sm.OLS(Y[col], X_fed).fit() for col in Y.columns}

with open("model2_summary.txt", "w") as f:
    f.write("Model 2: Federating Number vs Lexical Features\n")

for feature, model in model2.items():
    print(f"\nRegression Results for {feature}:\n")
    print(model.summary())
    with open("model2_summary.txt", "a") as f:
        f.write(f"\nRegression Results for {feature}:\n")
        f.write(model.summary().as_text())
        f.write("\n" + "-"*80 + "\n")


# #Model 3: Age versus Lexical Features

# # Make datetime.now() timezone-aware
now = datetime.now(timezone.utc)

# # Calculate the difference between 'birth' and now
merged_df['age_in_days'] = (now - merged_df['birth']).dt.days


# Define predictors (only log-transformed User Count)
X_age = merged_df[['age_in_days']]
X_age = sm.add_constant(X_age)  # Add intercept

# Fit regression models separately for each lexical feature
model3 = {col: sm.OLS(Y[col], X_age).fit() for col in Y.columns}

# # Print results
with open("model3_summary.txt", "w") as f:
    f.write("Model 3")

for feature, model in model3.items():
    print(f"\nRegression Results for {feature}:\n")
    print(model.summary())
    with open("model3_summary.txt", "a") as f:
        f.write(f"\nRegression Results for {feature}:\n")
        f.write(model.summary().as_text())
        f.write("\n" + "-"*80 + "\n")



#Model 4a: User Count and Federating Number versus Lexical Features
X_count_fed = merged_df[['Log_User_Count', 'Log_fed_num']]
X_count_fed = sm.add_constant(X_count_fed)  # Add intercept

# Fit regression models separately for each lexical feature
model4a = {col: sm.OLS(Y[col], X_count_fed).fit() for col in Y.columns}

with open("model4a_summary.txt", "w") as f:
    f.write("Model 4a")

for feature, model in model4a.items():
    print(f"\nRegression Results for {feature}:\n")
    print(model.summary())
    with open("model4a_summary.txt", "a") as f:
        f.write(f"\nRegression Results for {feature}:\n")
        f.write(model.summary().as_text())
        f.write("\n" + "-"*80 + "\n")

#Model 4b: User Count and Age versus Lexical Features   

X_count_age = merged_df[['Log_User_Count', 'age_in_days']]
X_count_age = sm.add_constant(X_count_age)  # Add intercept

model4b = {col: sm.OLS(Y[col], X_count_age).fit() for col in Y.columns}

with open("model4b_summary.txt", "w") as f:
    f.write("Model 4b")

for feature, model in model4b.items():
    print(f"\nRegression Results for {feature}:\n")
    print(model.summary())
    with open("model4b_summary.txt", "a") as f:
        f.write(f"\nRegression Results for {feature}:\n")
        f.write(model.summary().as_text())
        f.write("\n" + "-"*80 + "\n")

#Model 5: User Count, Federating Number, and Age versus Lexical Features

X = merged_df[['Log_User_Count', 'Log_fed_num', 'age_in_days']]
X = sm.add_constant(X)  # Add intercept

model5 = {col: sm.OLS(Y[col], X).fit() for col in Y.columns}

with open("model5_summary.txt", "w") as f:
    f.write("Model 5")

for feature, model in model5.items():
    print(f"\nRegression Results for {feature}:\n")
    print(model.summary())
    with open("model5_summary.txt", "a") as f:
        f.write(f"\nRegression Results for {feature}:\n")
        f.write(model.summary().as_text())
        f.write("\n" + "-"*80 + "\n")

