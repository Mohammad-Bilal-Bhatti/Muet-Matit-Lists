# Importing Required Libraries.
import pandas as pd


import os
cls = lambda:os.system('cls')


data_file_path = './/FilteredData.csv'

dataFrame = pd.read_csv(data_file_path)
features = ['District','U/R','CPN']
target = ['Discp.']


# ==================================================
# ======== New Static + Reversable Encoding ========
# ==================================================
# Prepare Map for Encoding or Decoding.


# Maping Dicplines
temp = dataFrame['Discp.'].value_counts()

tech_map = {}
start = 1

for idx in temp.index.values:
    tech_map[idx] = start
    start = start + 1


# Maping Districts
temp = dataFrame['District'].value_counts()
district_map = {}
start = 1

for idx in temp.index.values:
    district_map[idx] = start
    start = start + 1

# Maping U/R
temp = dataFrame['U/R'].value_counts()
ur_map = {}
start = 1

for idx in temp.index.values:
    ur_map[idx] = start
    start = start + 1


# ==================================================


# Encoding Categorical DATA.
# Dealing with categorical data...
# Applying real work...

for key in district_map:
    dataFrame['District'].loc[dataFrame['District'] == key] = district_map[key]

for key in ur_map:
    dataFrame['U/R'].loc[dataFrame['U/R'] == key] = ur_map[key]

for key in tech_map:
    dataFrame['Discp.'].loc[dataFrame['Discp.'] == key] = tech_map[key]


# Printing the Statistics
print(dataFrame.describe())


# Export the DataFrame to .csv
dataFrame.to_csv('EncodedData.csv', encoding='utf-8', header='true',index=False)
