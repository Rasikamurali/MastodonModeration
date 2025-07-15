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

#data 
df = pd.read_csv(r'data\regression_data_lexicalfeature_fed_birth.csv')
print(len(df))
print(df.columns)
df = df.rename(columns={'User Count_x': 'User Count'})

print(df.columns)
df['Log_User_Count'] = np.log10(df['User Count'])
print(len(df))


df = df[df['federating number'] > 0]  # Keep only positive values
df['Log_fed_num'] = np.log10(df['federating number'])
print(len(df))

births = (Counter(df['birth']))
# print(births.most_common(30))   

df = df[~df['birth'].str.contains("Failed to retrieve instance information. Status code: ", na=False)]

# Replace 'Unknown error' and 'Data not available' with NaN
df['birth'] = df['birth'].replace('Unknown error', np.nan)
df['birth'] = df['birth'].replace('Data not available', np.nan)

# Convert 'birth' column to datetime (coercing errors to NaT)
df['birth'] = pd.to_datetime(df['birth'], errors='coerce', utc=True)

# # Drop rows where 'birth' is NaT (invalid datetime values)
df = df.dropna(subset=['birth'])
print(len(df))

df = df.dropna(subset=['word count', 'rule count', 'TTR', 'Fk Score'])
print(len(df))



# # Define predictors (only log-transformed User Count)
X_User = df[['Log_User_Count']]
X_User = sm.add_constant(X_User)  # Add intercept

# Define multiple dependent variables (lexical features)
Y = df[['word count', 'rule count', 'TTR', 'Fk Score']]


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
X_fed = df[['Log_fed_num']]
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
df['age_in_days'] = (now - df['birth']).dt.days


# Define predictors (only log-transformed User Count)
X_age = df[['age_in_days']]
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
X_count_fed = df[['Log_User_Count', 'Log_fed_num']]
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

X_count_age = df[['Log_User_Count', 'age_in_days']]
X_count_age = sm.add_constant(X_count_fed)  # Add intercept

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

X = df[['Log_User_Count', 'Log_fed_num', 'age_in_days']]
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

