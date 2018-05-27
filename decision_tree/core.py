import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv(r'C:\Users\Сергеев\Documents\BDA_ML\decision_tree\train.csv', engine='python')
train = train.drop(["PassengerId", "Ticket","Cabin","Name"],axis=1)

train = pd.get_dummies(train) #Convert categorical variable into dummy/indicator variables

# Выбираем возраст nan, используем gradient boosting, чтобы обучиться на имеющихся данных, и просто предсказываем
nan_age = train[train.Age.apply(np.isnan)]
age = train[~train.Age.apply(np.isnan)]
gbr = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=1, loss='huber')
gbr = gbr.fit(age.drop("Age",axis=1), age["Age"])
nan_age["Age"] = gbr.predict(nan_age.drop("Age",axis=1))
train = age.append(nan_age)

data = train.drop("Survived",axis=1) 
target = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_score = tree_clf.predict(X_test)
print(roc_auc_score(y_test, y_score))

from myDecisionTree import MyDecisionTree
tree_clf = MyDecisionTree()
tree_clf.fit(X_train, y_train)
y_score = tree_clf.predict(X_test)
print(roc_auc_score(y_test, y_score))