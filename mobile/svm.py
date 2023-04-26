import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

input_file1 = r'C:\Users\user\Desktop\Machine Learning\mobile\train.csv'

dataset = pd.read_csv(input_file1, nrows=500)
x_train_dataset, y_train_dataset = dataset.iloc[:,:-1], dataset.iloc[:,-1]
x_train,x_test ,y_train, y_test = train_test_split(x_train_dataset,y_train_dataset,test_size=0.3)

#svm = OneVsRestClassifier(SVC(kernel='linear'))
svm = OneVsRestClassifier(SVC(kernel='rbf')) # gaussian

svm.fit(x_train, y_train) # ekpaideyetai to svm

scores = cross_val_score(svm, x_train, y_train, cv=10)
print("cross validation mean accuracy score: " + str(scores.mean()))

y_pred_svc = svm.predict(x_test) # vgainoyn oi prvlepseis panw sta x_test

svc_f1 = metrics.f1_score(y_test, y_pred_svc, average= "weighted") # f1_score
svc_accuracy = metrics.accuracy_score(y_test, y_pred_svc) # accuracy

print("F1 score: {}".format(svc_f1))
print("Accuracy score: {}".format(svc_accuracy))
