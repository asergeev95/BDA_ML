from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate = 0.02, 
                 iterations = 1000, 
                 epsilon = 0.1, 
                 threshold = 0.5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.threshold = threshold
        
    def fit(self, features, targets):
        self.train_features = features
        self.train_targets = targets
        self.weights = self.__calculate_weights()
    
    def predict(self, features):
        predictions = self.__predict_with_weights(features, self.weights) #get predicted probabilities
        return self.__classify(predictions) #round all probabilities to 0 or 1
    
    def __classify(self, predictions):
        return [(1 if prediction >= self.threshold else 0) for prediction in predictions]

    def __calculate_weights(self):
        weights = np.zeros(len(self.train_features[0]))
        i = 0 
        while(True): 
            improved_weights = self.__improve_weights(weights)
            if(i == self.iterations or np.linalg.norm(improved_weights-weights) < self.epsilon):
                break
            i = i+1
            weights = improved_weights
        return weights
    
    def __improve_weights(self, weights):
        predictions = self.__predict_with_weights(self.train_features, weights) #predict targets by current weights
        gradient = np.dot(self.train_features.T,  predictions - self.train_targets) #calculate gradient
        improved_weights = weights - self.learning_rate * gradient #get improved weights
        return improved_weights
    
    def __predict_with_weights(self, features, weights):
        z = np.dot(features, weights)
        return self.__logistic_function(z)
    
    def __logistic_function(self, x):
        return 1/(1+np.exp(-x))