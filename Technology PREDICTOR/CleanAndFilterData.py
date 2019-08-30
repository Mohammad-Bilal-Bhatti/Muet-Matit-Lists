# Importing Required Libraries.
import pandas as pd


import os
cls = lambda:os.system('cls')


data_file_path = './/Featured Data.csv'

dataFrame = pd.read_csv(data_file_path)
features = ['District','U/R','CPN']
target = ['Discp.']

# checking for the null or na values.
from prettytable import PrettyTable
    
prettyTable = PrettyTable()
prettyTable.field_names = ["Feature Name", "Sum(is-Null)"]
#Sum either any of the row has some null value in it?
for feature in features:
    summ = dataFrame[feature].isnull().sum()
    prettyTable.add_row([feature, summ])

print(prettyTable)

prettyTable = PrettyTable()
prettyTable.field_names = ["Feature Name", "Sum(Na)"]
#Sum either any of the row has some na values?
for feature in features:
    summ = dataFrame[feature].isna().sum()
    prettyTable.add_row([feature, summ])

print(prettyTable)


#count and print the different catagorical values.
print(dataFrame['District'].value_counts())
print(dataFrame['U/R'].value_counts())
print(dataFrame['Discp.'].value_counts())

# Filtering Data
dataFrame['District'] = dataFrame['District'].str.upper()
dataFrame['U/R'] = dataFrame['U/R'].str.upper()
dataFrame['Discp.'] = dataFrame['Discp.'].str.upper()

# Drop a row if it contains a certain value(either.g Discp.valuecount() <= 10)
temp = dataFrame['Discp.'].value_counts()

for idx in temp.index.values:
    count = temp.loc[idx]
    if(count <= 10):
        dataFrame=dataFrame[dataFrame['Discp.'] != idx ]

# Now again checking the different Disiplines.        
print(dataFrame['Discp.'].value_counts())


# Export the DataFrame to .csv
dataFrame.to_csv('FilteredData.csv', encoding='utf-8', header='true',index=False)
