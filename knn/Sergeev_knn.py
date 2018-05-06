import numpy as np
from sklearn import datasets
from collections import Counter

def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]

def distance(instance1, instance2):
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1) 
    instance2 = np.array(instance2)
    
    return np.linalg.norm(instance1 - instance2)
def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k):
    """
    get_neighors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with  
    (index, dist, label)
    where 
    index    is the index from the training_set, 
    dist     is the distance between the test_instance and the 
             instance training_set[index]
    distance is a reference to a function used to calculate the 
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

digits = datasets.load_digits()

digits_X = digits.data
digits_Y = digits.target

np.random.seed(0)
indices = np.random.permutation(len(digits_X))

digits_X_train = digits_X[indices[:-10]]
digits_y_train = digits_Y[indices[:-10]]
digits_X_test  = digits_X[indices[-10:]]
digits_y_test  = digits_Y[indices[-10:]]

for i in range(5):
    neighbors = get_neighbors(digits_X_train, 
                              digits_y_train, 
                              digits_X_test[i], 
                              5)
    print("index: ", i, 
          ", result of vote: ", vote(neighbors), 
          ", label: ", digits_y_test[i])