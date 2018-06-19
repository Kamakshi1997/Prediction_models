#!/usr/bin/python3
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#loading data
iris=load_iris()

train_data,test_data,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)
#calling KNN classifier
knn=KNeighborsClassifier(n_neighbors=5)

#loading data in knn
trained=knn.fit(train_data,train_target)

#printing original outputs
print(test_target)


#predicting model
output=trained.predict(test_data)
print(output)

#accuracy test
from sklearn.metrics import accuracy_score
pct=accuracy_score(test_target,output)
print(pct)
