import random
import numpy as np
import copy

class MyBaggingClassifier:
    def __init__(self, base_estimator, n_estimators =10, max_samples=1, max_features = 1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.base_estimators = [copy.deepcopy(self.base_estimator) for i in range(n_estimators)]
    def dataforest (self, data, labels):
            ind_row = sorted(random.sample(range(0, len(data)), self.max_samples))
            ind_column = sorted(random.sample(range(0, 10), self.max_features))
            d = data[ind_row]
            d = d[:,ind_column]
            return (d, labels[ind_row])
    def fit(self, X_train, Y_train):
        for i in range(0, self.n_estimators):
            data_p, labels_p = self.dataforest(X_train, Y_train)
            self.base_estimators[i].fit(data_p, labels_p)
        return
    
    def predict(self, X_test):
        pred = np.zeros((self.n_estimators,len(X_test)))
        answers = []
        for i in range(0, self.n_estimators):
            pred[i] = self.base_estimators[i].predict(X_test)
        pred = pred.transpose()
        for each in pred:
            answers.append(int(round(np.sum(each)/len(each))))
        return np.array(answers)