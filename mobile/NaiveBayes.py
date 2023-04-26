import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB

input_file1 = r'C:\Users\user\Desktop\Machine Learning\mobile\train.csv'

dataset = pd.read_csv(input_file1)
x_train_dataset, y_train_dataset = dataset.iloc[:,:-1], dataset.iloc[:,-1]
x_train,x_test ,y_train, y_test = train_test_split(x_train_dataset,y_train_dataset,test_size=0.3)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

scores = cross_val_score(gnb, x_train, y_train, cv=10)
print("cross validation mean accuracy score: " + str(scores.mean()))

y_pred_gnb=gnb.predict(x_test) #provlepseis sto x_test

gnb_accuracy = metrics.accuracy_score(y_test, y_pred_gnb) # accuracy
gnb_f1 = metrics.f1_score(y_test, y_pred_gnb, average = "weighted") # f1_score

print("F1 score: {}".format(gnb_f1))
print("Accuracy score: {}".format(gnb_accuracy))