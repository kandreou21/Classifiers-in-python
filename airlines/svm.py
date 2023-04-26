import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

input_file = r'C:\Users\user\Desktop\Machine Learning\airlines\airlines_delay.csv'

data = pd.read_csv(input_file, nrows=50000)

airline_encoding = dict((j,i) for i,j in enumerate(set(data['Airline'])))
airport_encoding = dict((j,i) for i,j in enumerate(set(data['AirportFrom']).union(set(data['AirportTo']))))
data['Airline'] = data['Airline'].map(airline_encoding)
data['AirportFrom'] = data['AirportFrom'].map(airport_encoding)
data['AirportTo'] = data['AirportTo'].map(airport_encoding)
df = pd.DataFrame(data)
newdf = data.drop('Flight',axis = 'columns')
x_train_dataset, y_train_dataset = newdf.iloc[:,:-1], newdf.iloc[:,-1]
x_train,x_test ,y_train, y_test = train_test_split(x_train_dataset,y_train_dataset,test_size=0.3)

svm = OneVsRestClassifier(SVC(kernel='linear'))
#svm = OneVsRestClassifier(SVC(kernel='rbf')) # gaussian

svm.fit(x_train, y_train) # ekpaideyetai to svm

y_pred_svc = svm.predict(x_test) # vgainoyn oi prvlepseis panw sta x_test

svc_f1 = metrics.f1_score(y_test, y_pred_svc, average= "weighted") # f1_score
svc_accuracy = metrics.accuracy_score(y_test, y_pred_svc) # accuracy

print("F1 score: {}".format(svc_f1))
print("Accuracy score: {}".format(svc_accuracy))