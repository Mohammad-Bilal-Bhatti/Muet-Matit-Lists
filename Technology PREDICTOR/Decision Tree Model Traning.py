# Importing ML Libraries.
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import seaborn as sn

import os
cls = lambda: os.system('cls')

data_file_path = './/FilteredData.csv'

dataFrame = pd.read_csv(data_file_path)
features = ['District','U/R','CPN']
target = ['Discp.']

# ==================================================
# ======== New Static + Reversable Encoding ========
# ==================================================
# Prepare Map for Encoding or Decoding.

temp = dataFrame['Discp.'].value_counts()

tech_map = {}
start = 1
# Maping Dicplines
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



print('========== Data Frame =========')
print(dataFrame)
print('======== Data Frame End =======')
print('\n')

print('=========== Statistics ============')
print(dataFrame.describe())    
print('========= Statistics End ===========')
print('\n')



X = dataFrame[features]
Y = dataFrame[target]

# Split dataset into training set and test set
# 75% training and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25) 


tech_names = list(tech_map.keys())

#Creating Decision Tree Modal
# Create Decision Tree classifer object

# Doing Some Calculations.
import math
total_tech = len(tech_names)
pref_depth = math.ceil(math.log2(total_tech))



clf = DecisionTreeClassifier(criterion="entropy", max_depth=pref_depth)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# ======== MODEL EVALUATION ===========
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print('\n')

from sklearn import metrics
import numpy as np
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

print('Root Mean Squared Error: ',rmse)
print('\n')



cfm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cfm, annot=True)
plt.show()
# ======== END MODEL EVALUATION ===========

    

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features,class_names=tech_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dendogramwithdepth.png')
Image(graph.create_png())



for depth in range(1,31):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Depth: ",depth," Accuracy:",metrics.accuracy_score(y_test, y_pred))



# Max Accuracy is at 8 depth

clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# ======== MODEL EVALUATION ===========


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print('\n')

from sklearn import metrics
import numpy as np
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print('Root Mean Squared Error: ',rmse)
print('R2 Score :',r2)
print('\n')



cfm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cfm, annot=True)
plt.show()


# ======== END MODEL EVALUATION ===========



from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features,class_names=tech_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dendogramwithdepth8.png')
Image(graph.create_png())


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
    result = clf.predict(testData)[0]
    mTech = getKey(tech_map,result)
    print('Based On Your Results You Could Be Awarded: ',mTech,' Technology')
    again = input('Again (Y/N)? ')
    again = again.lower()
    again = True if again == 'y' else False


# ================ MY TEST END ==================
