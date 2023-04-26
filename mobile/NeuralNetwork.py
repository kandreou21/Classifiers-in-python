import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

input_file1 = r'C:\Users\user\Desktop\Machine Learning\mobile\train.csv'

dataset = pd.read_csv(input_file1)
x_train_dataset, y_train_dataset = dataset.iloc[:,:-1], dataset.iloc[:,-1]
x_train,x_test ,y_train, y_test = train_test_split(x_train_dataset,y_train_dataset,test_size=0.3)

#neurons in hidden layers
K = 200
K1 = 200
K2 = 100

# create a neural network with one hidden layer and K hidden neurons
model = MLPClassifier(hidden_layer_sizes=(K,), activation='logistic', solver='sgd')
#model = MLPClassifier(hidden_layer_sizes=(K1,K2), activation='logistic', solver='sgd', max_iter=300)

scores = cross_val_score(model, x_train, y_train, cv=10)
print("cross validation mean accuracy score: " + str(scores.mean()))

# set the activation function for the output layer to softmax
model.out_activation_ = 'softmax'
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

gnb_accuracy = metrics.accuracy_score(y_test, y_pred) # accuracy
gnb_f1 = metrics.f1_score(y_test, y_pred, average="weighted") # f1_score

print("F1 score: {}".format(gnb_f1))
print("Accuracy score: {}".format(gnb_accuracy))