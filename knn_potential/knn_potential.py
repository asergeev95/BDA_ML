import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from knnPotentialClassifier import PotentialKnn

digits = datasets.load_digits()
(X_train, X_test, y_train, y_test) = train_test_split(digits.data, digits.target, test_size=0.25)
N = X_train.shape[0]
model = PotentialKnn(3, N)
print("====FIT====")
model.fit(X_train, y_train)
print("===SCORE===")
model.score(X_test, y_test) 
confusion_matrix(model.predict(X_test), y_test)