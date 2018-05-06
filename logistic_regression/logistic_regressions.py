from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from MyLogisticRegression import MyLogisticRegression

def calculate_accuracy(predicted, targets):
    accuracy = 0
    for predicted_value, target in zip(predicted, targets):
        if predicted_value == target:
            accuracy += 1
    return accuracy/len(predicted)

cancer = load_breast_cancer()
train_targets, test_targets, train_datas, test_datas =  train_test_split(cancer.target, cancer.data, test_size = 0.2, random_state=5)

cp = MyLogisticRegression() 
cp.fit(train_datas, train_targets) 
predicted = cp.predict(test_datas) 
print("My accuracy:", calculate_accuracy(predicted, test_targets))

sklg = LogisticRegression()
sklg.fit(train_datas, train_targets)
sklg_predicted = sklg.predict(test_datas)
print("Sklearn accuracy:",calculate_accuracy(sklg_predicted, test_targets))