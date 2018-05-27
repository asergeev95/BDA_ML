from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np

def dist(a, b):
    d = 0.0
    for i in range(len(a)):
        d += (a[i] - b[i])**2
    return d**0.5

class PotentialKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, k, N):
        print(k, N)
        self.weights = np.zeros((N))
        self.k = k
        self.N = N

    def fit(self, X_train, y_train):
        print('fit')
        iter = 0
        self.x = X_train
        self.y = y_train
        predictions = self.predict(X_train) # инициализация предсказаний, потом уже 
                                            # для каждой точки делаем предсказание, 
                                            # пока оно не совпадет с тем, которое реально реализуется в этой точке
        while self.score(X_train, y_train) < 0.9:
            iter += 1
            print('iteration= {0}'.format(iter))
            for _ in range(100): #обрабатываем сразу 100 точек, иначе очень долго считается
                i = np.random.randint(self.N)   #Берем рандомную точку из выборки и проверяем, 
                                                # совпадет ли ее реальный класс с предсказуемым. 
                                                # Если нет, то пересчитываем веса
                if predictions[i] != y_train[i]:
                    self.weights[i] += 1 
            predictions = self.predict(X_train) 
        return self.weights 

    def predict(self, test_data):
        print('predict')
        listofpred = []
        k = self.k
        for test_point in (test_data):
            j = 0
            d = [[dist(test_point, point), self.y[ind]]
                 for ind, point in enumerate(self.x)]
            stat = [0 for _ in range(10)]
            for z in sorted(d)[0:k]:
                j += 1
                stat[z[1]] += self.weights[j] * 1 / (z[0] + 1) #Функция взвешанного knn
            listofpred.append(sorted(zip(stat, range(10)), reverse=True)[0][1]) 
            # Берем 10 классов, сортируем значение функции взвешенного КНН и получаем чиселку - нужный класс от 0 до 9
        return listofpred
