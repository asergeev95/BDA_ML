import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()

digits_X = digits.data
digits_Y = digits.target

np.random.seed(0)
indices = np.random.permutation(len(digits_X))

digits_X_train = digits_X[indices[:-10]]
digits_y_train = digits_Y[indices[:-10]]
digits_X_test  = digits_X[indices[-10:]]
digits_y_test  = digits_Y[indices[-10:]]

knn = KNeighborsClassifier()
knn.fit(digits_X_train, digits_y_train) 

y_predicted = knn.predict(digits_X_test)

confusion_matrix(y_predicted, digits_y_test)