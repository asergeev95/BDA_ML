import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection

dataset = pd.read_csv('house_price_predict.csv')

X_train_initial, X_test_initial, Y_train, Y_test = sk.model_selection.train_test_split(
    dataset, dataset["Price"].values, test_size=0.33, random_state=5)

X_train = X_train_initial.drop("Price", axis=1).fillna(0)
X_test = X_test_initial.drop("Price", axis=1).fillna(0)

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_predicted = lm.predict(X_test)

X_train = X_train_initial.drop("Price", axis=1).fillna(X_train_initial.mean())
X_test = X_test_initial.drop("Price", axis=1).fillna(X_test_initial.mean())

lm.fit(X_train, Y_train)

Y_predicted_new = lm.predict(X_test)

mseFull = np.mean((Y_test - Y_predicted)**2)
mseFull_new = np.mean((Y_test - Y_predicted_new)**2)

##Засунуть всё в цикл, изменять test_size с неким шагом, построить табличку
# test_size — mseFull — mseFull_new 