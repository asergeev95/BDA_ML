from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class MyDecisionTree(BaseEstimator, ClassifierMixin):
    class Node:
        def __init__(self, column_name, column_value, result, left = None, right = None):
            self.column_name = column_name
            self.column_value = column_value
            self.result = result
            self.left = left
            self.right = right
        
    def fit(self, X, y):
        self.tree = self.__split(X, y)
        return self

    def predict(self, X):
        result = []
        for row in range(0, len(X)):
            result.append(self.__predict(X, row, self.tree))
        return result
    
    def visualize(self):
        self.__visualize(0, 'root', '', '', self.tree)
        
    def __split(self, X, y):
        best_val, best_column, best_cost = None, None, 1000
        for col in X.columns:
            val, cost = self.__split_column(X, y, col)
            if best_cost > cost:
                best_cost = cost
                best_column = col
                best_val = val

        left_node, right_node = self.__split_by_value(X, best_column, best_val)
        leftX, leftY = X[left_node].drop(best_column, axis=1), y[left_node]
        rightX, rightY = X[right_node].drop(best_column, axis=1), y[right_node]
        
        if len(leftX) > 0 and len(rightX) > 0:
            l = self.__split(leftX, leftY)
            r = self.__split(rightX, rightY)
            return MyDecisionTree.Node(best_column, best_val,None, l, r)
        elif len(leftX) > 0:
            return MyDecisionTree.Node(best_column, best_val, leftY.ravel()[0])
        else: 
            return MyDecisionTree.Node(best_column, best_val, rightY.ravel()[0])
       
    def __split_column(self, X, y, column_name):
        best_value, best_cost = None, 1000
        for val in self.__enumerate_split_points(X[column_name]):
            cost = self.__calc_cost_for_value(X, y, column_name, val)
            
            if cost == 0:
                return (val, cost)
            elif cost < best_cost:
                best_cost = cost
                best_value = val
        return (best_value, best_cost)
    
    def __enumerate_split_points(self, col):
        unique = col.unique().ravel()
        unique = np.sort(unique) 
        if len(unique) == 1:
            yield unique[0]
        else:
            for index in range(1, len(unique)):
                prev = unique[index - 1]
                current = unique[index]
                yield (prev + current)/2
            
    def __calc_cost_for_value(self, X, y, column_name, value):
        left_node, right_node = self.__split_by_value(X, column_name, value)
        left  = X[left_node]
        right = X[right_node]
        
        ll = y[left_node][y == 1]
        lr = y[left_node][y == 0]

        rl = y[right_node][y == 1] 
        rr = y[right_node][y == 0]
        
        lgini = self.__gini_index(np.array([len(ll), len(lr)]))
        rgini = self.__gini_index(np.array([len(rl), len(rr)]))

        total = len(left) + len(right)
        return lgini*len(left)/total + rgini*len(right)/total
    
    def __split_by_value(self, X, column, value):
        return (X[column] <= value, X[column] > value)
        
    def __gini_index(self, splits):
        total = sum(splits)
        if total == 0:
            return 0
        return 1 - sum((splits/total)**2)
    
    def __visualize(self, depth, text, sign, value, node):
        if node is None:
            return
        
        res = '' if node.result is None else '(' + str(node.result) + ')'
        print('--'*depth, str(depth) + ')', text, sign, value, res)
        self.__visualize(depth + 1, node.column_name, '<=', node.column_value, node.left)
        self.__visualize(depth + 1, node.column_name, '>', node.column_value, node.right)
        
    def __predict(self, X, row, node):
        if node.result is not None:
            return node.result

        if X.iloc[row][node.column_name] <= node.column_value:
            return self.__predict(X, row, node.left)
        
        return self.__predict(X, row, node.right)