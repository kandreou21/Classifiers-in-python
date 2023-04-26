import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate


input_file = r'C:\Users\user\Desktop\Machine Learning\airlines\airlines_delay.csv'

data = pd.read_csv(input_file)

airline_encoding = dict((j,i) for i,j in enumerate(set(data['Airline'])))
airport_encoding = dict((j,i) for i,j in enumerate(set(data['AirportFrom']).union(set(data['AirportTo']))))
data['Airline'] = data['Airline'].map(airline_encoding)
data['AirportFrom'] = data['AirportFrom'].map(airport_encoding)
data['AirportTo'] = data['AirportTo'].map(airport_encoding)
df = pd.DataFrame(data)
newdf = data.drop('Flight',axis = 'columns')
x_train_dataset, y_train_dataset = newdf.iloc[:,:-1], newdf.iloc[:,-1]
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
gnb_f1 = metrics.f1_score(y_test, y_pred, average = "weighted") # f1_score

print("F1 score: {}".format(gnb_f1))
print("Accuracy score: {}".format(gnb_accuracy))