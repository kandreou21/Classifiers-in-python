import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

input_file1 = r'C:\Users\user\Desktop\Machine Learning\mobile\train.csv'

dataset = pd.read_csv(input_file1)
x_train_dataset, y_train_dataset = dataset.iloc[:,:-1], dataset.iloc[:,-1]
x_train,x_test ,y_train, y_test = train_test_split(x_train_dataset,y_train_dataset,test_size=0.3)

k = [1,3,5,10]

for neighbor in k:
    knn = KNeighborsClassifier(n_neighbors = neighbor, metric = 'minkowski', p=2)
    knn.fit(x_train, y_train) # ekpaideyetai o knn
    scores = cross_val_score(knn, x_train, y_train, cv=10)
    print(str(neighbor) + " neighbors:")
    print("cross validation mean accuracy score: " + str(scores.mean()))

    y_pred_knn = knn.predict(x_test) # vgainoyn oi prvlepseis panw sta x_test

    knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn) # accuracy
    knn_f1 = metrics.f1_score(y_test, y_pred_knn, average = "weighted") # f1_score

    print("F1 score: {}".format(knn_f1))
    print("Accuracy score: {}".format(knn_accuracy))

