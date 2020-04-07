# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:26:10 2019

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt

pdata = pd.read_csv('Dataset.csv')

pdata.shape

pdata.head()

pdata = pdata.drop("Type",axis=1)

pdata.head()

pdata.isnull().values.any()

pdata.corr()

n_true = len(pdata.loc[pdata['Class'] == True])
n_false = len(pdata.loc[pdata['Class'] == False])
print("Number of true cases: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))
print("Number of false cases: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))


from sklearn.model_selection import train_test_split

features_cols = ['Age', 'BS Fast', 'BS pp', 'Plasma R', 'Plasma F', 'HbA1c']
predicted_class = ['Class']

X = pdata[features_cols].values     # Predictor feature columns (6 X m)
Y = pdata[predicted_class].values   # Predicted class (1=True, 0=False) (1 X m)
split_test_size = 0.30
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split_test_size,random_state=0)

print("{0:0.2f}% data is in training set".format((len(x_train)/len(pdata.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(x_test)/len(pdata.index)) * 100))

print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['Class'] == 1]), (len(pdata.loc[pdata['Class'] == 1])/len(pdata.index)) * 100))
print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['Class'] == 0]), (len(pdata.loc[pdata['Class'] == 0])/len(pdata.index)) * 100))
print("")
print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))
print("")
print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
print("")


df_x_train = pd.DataFrame(x_train, columns = ['Age', 'BS Fast', 'BS pp', 'Plasma R', 'Plasma F', 'HbA1c'])
df_x_test = pd.DataFrame(x_test, columns = ['Age', 'BS Fast', 'BS pp', 'Plasma R', 'Plasma F', 'HbA1c'])
df_x_train.head()

from sklearn.preprocessing import Imputer

rep_0 = Imputer(missing_values=0, strategy="mean", axis=0)

x_train = rep_0.fit_transform(x_train)
x_test = rep_0.fit_transform(x_test)

df_x_train_after = pd.DataFrame(x_train, columns = ['Age', 'BS Fast', 'BS pp', 'Plasma R', 'Plasma F', 'HbA1c'])
df_x_test_after = pd.DataFrame(x_test, columns = ['Age', 'BS Fast', 'BS pp', 'Plasma R', 'Plasma F', 'HbA1c'])
df_x_train_after.head()

# Re-scaling dataset:
# MinMaxScaler transforms features by scaling each feature to a given range.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


#Implementing SVM
from sklearn.svm import SVC
svc = SVC( gamma = 'auto',random_state=0)
svc.fit(x_train, y_train.ravel())
svc_predict_train = svc.predict(x_train)
from sklearn import metrics
#get accuracy
svc_accuracy = metrics.accuracy_score(y_train, svc_predict_train)
#print accuracy
print ("Accuracy of training set: {0:.4f}".format(svc_accuracy))

svc_predict_test = svc.predict(x_test)

#get accuracy
svc_accuracy_testdata = metrics.accuracy_score(y_test, svc_predict_test)
#print accuracy
print ("Accuracy of test set: {0:.4f}".format(svc_accuracy_testdata))

print ("Confusion Matrix for SVM")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, svc_predict_test, labels=[1, 0])))
print ("")




#Implementing RandomForestAlgorithm

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train_scaled, y_train.ravel())
rf_predict_train = rf.predict(x_train_scaled)
from sklearn import metrics
#get accuracy
rf_accuracy = metrics.accuracy_score(y_train, rf_predict_train)
#print accuracy
print ("Accuracy of trainng set: {0:.4f}".format(rf_accuracy))
rf_predict_test = rf.predict(x_test)
#get accuracy
rf_accuracy_testdata = metrics.accuracy_score(y_test, rf_predict_test)

#print accuracy
print ("Accuracy of test set: {0:.4f}".format(rf_accuracy_testdata))

print ("Confusion Matrix for Random Forest")
# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0])))
print ("")



#Implementing LogisticRegression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
logreg.fit(x_train, y_train)
lg_predict_train = logreg.predict(x_train_scaled)

from sklearn import metrics
#get accuracy
lg_accuracy = metrics.accuracy_score(y_train, lg_predict_train)
#print accuracy
print ("Accuracy of training set: {0:.4f}".format(rf_accuracy))
lg_predict_test = logreg.predict(x_test_scaled)
#get accuracy
lg_accuracy_testdata = metrics.accuracy_score(y_test, lg_predict_test)

#print accuracy
print ("Accuracy of test set: {0:.4f}".format(lg_accuracy_testdata))
print ("Confusion Matrix for Logistic Regression")
# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, lg_predict_test, labels=[1, 0])))
print ("")



#Implementing Graph
x=['SVM','R.F','L.R']
y=[svc_accuracy_testdata,rf_accuracy_testdata,lg_accuracy_testdata]
a=svc_accuracy_testdata*100
b=rf_accuracy_testdata*100
c=lg_accuracy_testdata*100
plt.bar(x,y,color=("red","green","blue"))
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparision')
Result={"SVM":a,"R.F":b,"L.R":c}
print(Result)




