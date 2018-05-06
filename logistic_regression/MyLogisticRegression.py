from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate = 0.01, 
                 iterations = 5000, 
                 epsilon = 0.1, 
                 threshold = 0.5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.threshold = threshold
        
    def fit(self, features, targets):
        self.train_features = features
        self.train_targets = targets
        self.weights = self.calculate_weights()
    
    def predict(self, features):
        predictions = self.predict_with_weights(features, self.weights) 
        return self.classify(predictions)
    
    def classify(self, predictions):
        return [(1 if prediction >= self.threshold else 0) for prediction in predictions]

    def calculate_weights(self):
        weights = np.zeros(len(self.train_features[0]))
        i = 0 
        while(True): 
            improved_weights = self.improve_weights(weights)
            if(i == self.iterations or np.linalg.norm(improved_weights-weights) < self.epsilon):
                break
            i = i+1
            weights = improved_weights
        return weights
    
    def improve_weights(self, weights):
        predictions = self.predict_with_weights(self.train_features, weights) #predict targets by current weights
        gradient = np.dot(self.train_features.T,  predictions - self.train_targets) #calculate gradient
        improved_weights = weights - self.learning_rate * gradient #get improved weights
        return improved_weights
    
    def predict_with_weights(self, features, weights):
        z = np.dot(features, weights)
        return self.logistic_function(z)
    
    def logistic_function(self, x):
        return 1/(1+np.exp(-x))