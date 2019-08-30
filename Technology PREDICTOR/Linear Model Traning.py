# Importing ML Libraries.
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import mean_squared_error, r2_score


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

temp = dataFrame['District'].value_counts()

district_map = {}
# Maping Districts
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


print('========== Data Frame =========')
print(dataFrame)
print('======== Data Frame End =======')
print('\n')

print('=========== Statistics ============')
print(dataFrame.describe())    
print('========= Statistics End ===========')
print('\n')

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


# Displaying the Data Frame

print('========== Data Frame =========')
print(dataFrame.head())
print('======== Data Frame End =======')
print('\n')

print('=========== Statistics ============')
print(dataFrame.describe())    
print('========= Statistics End ===========')
print('\n')

# ==================================================


X = dataFrame[features]
Y = dataFrame[target]




# Create Test/Train
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)

# Scikit Logistic Regression 
scikit_log_reg = LogisticRegression(solver='lbfgs',multi_class='multinomial' ,max_iter=20000)
scikit_log_reg.fit(x_train, y_train.values.reshape(-1,))



y_pred = scikit_log_reg.predict(x_test[features])


#Evaluation of the results.
from sklearn import metrics
print('Algorithm Accuracy: ',metrics.accuracy_score(y_test,y_pred))
print('\n')

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print('Root Mean Squared Error: ',rmse)
print('R2 Score :',r2)
print('\n')



cfm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cfm, annot=True)
plt.show()
# print(cfm)


# ================ MY TEST ====================

# Displaying the Data Frame
print('============ District =============')
print(district_map)
print('========== District End ===========')
print('\n')

print('========== Urban Rural ===========')
print(ur_map)
print('======== Urban Rural End =========')
print('\n')

print('============ Discpline =============')
print(tech_map)
print('========== Discpline End ===========')
print('\n')

def getKey(dictionary, item):
    for key in dictionary:
        if dictionary[key] == item:
            return key
    return None

again = True
while again:
    mDistrict = input('Enter District: ').upper()
    mUR = input('Urban Or Rural U/R? ').upper()
    mCPN = float(input('Enter CPN: '))
    testData = np.array([[district_map[mDistrict],ur_map[mUR],mCPN]])
    result = scikit_log_reg.predict(testData)[0]
    mTech = getKey(tech_map,result)
    print('Based On Your Results You Could Be Awarded: ',mTech,' Technology')
    again = input('Again (Y/N)? ')
    again = again.lower()
    again = True if again == 'y' else False


# ================ MY TEST END ==================











