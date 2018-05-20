import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingRegressor
from myDecisionTree import MyDecisionTree

train = pd.read_csv('train.csv')
train = train.drop(["PassengerId", "Ticket","Cabin","Name"],axis=1)

train = pd.get_dummies(train)

nan_age = train[train.Age.apply(np.isnan)]
age = train[train.Age.apply(np.isnan)]
est = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=1, loss='huber')
est = est.fit(age.drop("Age",axis=1), age["Age"])
nan_age["Age"] = est.predict(nan_age.drop("Age",axis=1))
train = age.append(nan_age)

data, target = train.drop("Survived",axis=1), train['Survived']
(X_train, X_test, y_train, y_test) = train_test_split(data, target, test_size=0.2)


tree_clf  = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_score = tree_clf.predict(X_test)
roc_auc_score(y_test, y_score)

# tree_clf = MyDecisionTree()
# tree_clf.fit(X_train, y_train)
# tree_clf.visualize()
# y_score = tree_clf.predict(X_test)
# roc_auc_score(y_test, y_score)