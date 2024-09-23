import pandas as pd 
import numpy as np 

data_1_15 = pd.read_csv(r'data\instance rules\instance_rules_1_15.csv')
data_16_35 = pd.read_csv(r'data\instance rules\instance_rules_16_35.csv')
data_36_150 =pd.read_csv(r'data\instance rules\instance_rules_36_150.csv')
data_151_500= pd.read_csv(r'data\instance rules\instance_rules_151_500.csv')
data_501_1500 = pd.read_csv(r'data\instance rules\instance_rules_501_1500.csv')
data_1501_5000 = pd.read_csv(r'data\instance rules\instance_rules_1501_5000.csv')
data_5001= pd.read_csv(r'data\instance rules\instance_rules_5001.csv')

complete_instance_data = pd.concat([data_1_15, data_16_35, data_36_150, data_151_500, data_501_1500, data_1501_5000, data_5001])


print(complete_instance_data.columns)

# Define the bins and labels for the categories
bins = [0, 5, 15, 50, 150, 500, 1500, 5000, float('inf')]
labels = ['1 to 5', '6 to 15', '16 to 50', '51 to 150', '151 to 500', '501 to 1500', '1501 to 5000', '5000+']

# Create and add the 'category' column to main_df
complete_instance_data['instance group'] = pd.cut(complete_instance_data['User Count'], bins=bins, labels=labels, right=True, include_lowest=True)

print(complete_instance_data.head())

complete_instance_data.to_csv('merged_instance_data.csv')