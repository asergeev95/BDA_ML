import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('lab7_data.csv', sep='\t', decimal=',')
data_targets, prediction_data = data["churn"], data.drop("churn", axis=1)
train_targets, test_targets, train_data, test_data = train_test_split(data_targets, prediction_data, test_size = 0.33, random_state=4)

knn_parameters = {'n_neighbors': range(1, 120)}
knn = GridSearchCV(KNeighborsClassifier(), knn_parameters, scoring='roc_auc')
print(knn)
knn.fit(train_data, train_targets)
knn_predicted = knn.predict(test_data)
print(roc_auc_score(test_targets, knn_predicted))

svm_parameters = [{'C': [0.1, 1, 10], 'kernel': ['linear']},{'C': [0.1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
#svm_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = GridSearchCV(SVC(), svm_parameters, verbose=2, scoring='roc_auc')
svc.fit(train_data, train_targets) 
svc_predicted = svc.predict(test_data)
print(roc_auc_score(test_targets, svc_predicted))

decision_tree_params={'min_samples_split': range(2, 10), 'max_depth': range(2, 10)}
dtc = GridSearchCV(DecisionTreeClassifier(), decision_tree_params, scoring='roc_auc')
dtc.fit(train_data, train_targets)
dtc_predicted = dtc.predict(test_data)
print(roc_auc_score(test_targets, dtc_predicted))