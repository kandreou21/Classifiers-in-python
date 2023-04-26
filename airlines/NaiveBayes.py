import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB

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


gnb = GaussianNB()
gnb.fit(x_train, y_train)

scores = cross_val_score(gnb, x_train, y_train, cv=10)
print("cross validation mean accuracy score: " + str(scores.mean()))

y_pred_gnb=gnb.predict(x_test)

gnb_accuracy = metrics.accuracy_score(y_test, y_pred_gnb) # accuracy
gnb_f1 = metrics.f1_score(y_test, y_pred_gnb, average = "weighted") # f1_score

print("F1 score: {}".format(gnb_f1))
print("Accuracy score: {}".format(gnb_accuracy))