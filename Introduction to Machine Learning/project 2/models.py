import numpy as np
import pandas as pd
from collections import Counter
import toolkit as tk

"""
Helper functions
"""
# determine the class of a datapoint given all its neighbors
def determine_class(x, neighbor_y, neighbor_x, gamma, cyclic):
    # If gamma is not None, we have a regression problem
    if gamma:
        weights = gamma_kernel(x, neighbor_x, gamma, cyclic)
        return np.sum(neighbor_y * weights) / np.sum(weights)
    # If gamma is not None, we have a classification problem
    else: 
        return plurality_vote(neighbor_y)

# for classification problem
# employ a plurality vote to determine the class
def plurality_vote(neighbors):
    vote_counts = Counter(neighbors)
    return vote_counts.most_common(1)[0][0]

# for regression problem
# employ a Gaussian (radial basis function) kernel
def gamma_kernel(x, xq, gamma, cyclic):
    # K(x,xq) = exp(−γ||x − xq||2)
    distences = euclidean_distances(x, xq, cyclic)
    weights = np.exp(-gamma * np.array(distences))
    return weights

# determine the distances of data point x to the training data points
# cyclic = {feature index with cyclic: the cyclic range}
def euclidean_distances(x, X_train, cyclic):
    if len(x) != len(X_train[0]):
        # print("x: ", x)
        # print("X_train: ", X_train)
        raise ValueError("Vectors must have the same dimensionality")
    distances = []
    # calculate euclidean distance for each X_train to x
    for x_train in X_train:
        squared_diff = []
        for i in range(len(x)): 
            if i not in cyclic.keys():
                # not a cyclic feature, use linag norm
                squared_diff.append((x[i] - x_train[i])**2)
            else:
                # cyclic feature, use circular Euclidean distance
                diff = abs(x[i] - x_train[i])
                if diff > cyclic[i]/2:
                    diff = cyclic[i] - diff
                squared_diff.append(diff**2)
                
        distance = np.sqrt(sum(squared_diff))
        distances.append(distance)
    return distances

# find the k nearest neighbor using linalg norm
def get_neighbors(X_train, y_train, x, k, cyclic, exclude=False):
    if exclude:
        # find removing index
        r_i = np.where(np.all(X_train==x,axis=1))
        X_train = np.delete(X_train, r_i, 0)
        y_train = np.delete(y_train, r_i)
    distances = euclidean_distances(x, X_train, cyclic)
    sorted_indices = np.argsort(distances)
    neighbor_y = [y_train[i] for i in sorted_indices[:k]]
    neighbor_x = [X_train[i] for i in sorted_indices[:k]]
    return neighbor_y, neighbor_x

"""
K-Nearest Neighbor
When parameter gamma is used to initialize the KNN
indicating this is a regression problem
Example usage:
knn_regressor = KNN(k=5, gamma=0.1)  # for regression
knn_regressor.fit(X_train, y_train)
y_pred_regression = knn_regressor.predict(X_test)

knn_classifier = KNN(k=5)  # for classification
knn_classifier.fit(X_train, y_train)
y_pred_classification = knn_classifier.predict(X_test)
"""
class KNN:
    def __init__(self, k=3, gamma=None, cyclic={}):
        self.k = k
        self.gamma = gamma
        self.cyclic = cyclic
        
    # no specific things done for fit here, just storing x and y
    # the cyclic determines if there are cyclic features, default to be all False
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    # predict the data using the neighbors
    def predict(self, X_test):
        predictions = []
        neighbors = []
        for x in X_test:
            neighbor_y, neighbor_x = get_neighbors(self.X_train, self.y_train, x, self.k, self.cyclic)
            neighbors.append([neighbor_y, neighbor_x])
            predictions.append(determine_class(x, neighbor_y, neighbor_x, self.gamma, self.cyclic))
        return neighbors, predictions
    
    # this function requires the hyperparameter to be passed in as dictionary
    def update_param(self, param_setting):
        self.k = param_setting['k']
    
"""
Edited K-Nearest Neighbor 
When parameter gamma is used to initialize the Edited KNN
indicating this is a regression problem
The edition stops when the performance starts to decrease
Example usage:
knn_regressor = EditedKNN(k=5, epsilon=0.1, gamma=0.1)  # for regression
knn_regressor.fit(X_train, y_train)
y_pred_regression = knn_regressor.predict(X_test)

knn_classifier = EditedKNN(k=5, epsilon=0.1)  # for classification
knn_classifier.fit(X_train, y_train)
y_pred_classification = knn_classifier.predict(X_test)
"""
class EditedKNN:
    def __init__(self, k=3, epsilon=0.1, gamma=None, cyclic={}):
        self.k = k
        self.epsilon = epsilon
        self.gamma = gamma
        self.cyclic = cyclic

    # we passin the X_test and y_test as well to determine when to stop editing
    # calculate the edited datasets 
    def fit(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._edit_dataset()

    def predict(self, X_test):
        predictions = []
        neighbors = []
        for x in X_test:
            neighbor_y, neighbor_x = get_neighbors(self.X_train, self.y_train, x, self.k, self.cyclic)
            neighbors.append([neighbor_y, neighbor_x])
            predictions.append(determine_class(x, neighbor_y, neighbor_x, self.gamma, self.cyclic))
        return neighbors, predictions

    def update_param(self, param_setting):
        self.k = param_setting['k']
        if 'epsilon' in param_setting.keys():
            self.epsilon = param_setting['epsilon']
           
    def _is_classified_correctly(self, pred, test):
        if self.gamma:
            # “correct” prediction if the prediction falls within some ε of the ground truth value
            if pred <= test + self.epsilon and pred >= test - self.epsilon:
                return True
        else:
            if pred == test:
                return True
        return False
    
    # calculate the performance after the data removal using KNN
    def _performance(self, X_train, y_train):
        knn = KNN(k=self.k, gamma=self.gamma) 
        knn.fit(X_train, y_train)
        _, y_pred = knn.predict(self.X_test)
        if not self.gamma:
            # Use classification metric (accuracy)
            performance_metric = tk.evaluation_metric(self.y_test, y_pred, task_type='classification')
            performance = performance_metric['accuracy']
        else:
            # Use regression metric (r2)
            # evaluate the performance based on the variation the regression covers
            performance_metric = tk.evaluation_metric(self.y_test, y_pred, task_type='regression')
            performance = performance_metric['r2']
        return performance
    
    def _edit_dataset(self):
        # Initial dataset size
        initial_size = len(self.X_train)

        # Create a list to mark data points for deletion (True for deletion)
        to_delete = [False] * initial_size
        
        # record initial performance
        performance = self._performance(self.X_train, self.y_train)
        
        # store a temporary edited dataset
        tmp_edited_X = self.X_train
        tmp_edited_y = self.y_train
        
        for i in range(initial_size):
            if not to_delete[i]:  # Skip data points already marked for deletion
                x = self.X_train[i]
                y = self.y_train[i]
                # use its single nearest neighbor to make a prediction
                # doing prediction manually as we need to exlude i
                neighbor_y, neighbor_x = get_neighbors(tmp_edited_X, tmp_edited_y, x, k=1, exclude=True, cyclic=self.cyclic)
                # get the prediction of that x using determine class
                prediction = determine_class(x, neighbor_y, neighbor_x, self.gamma, self.cyclic)

                # Check if the prediction is incorrect
                if not self._is_classified_correctly(prediction, y):
                    to_delete[i] = True
                    # if deleted, check the performance
                    tmp_edited_X = [self.X_train[i] for i in range(initial_size) if not to_delete[i]]
                    tmp_edited_y = [self.y_train[i] for i in range(initial_size) if not to_delete[i]]
                    cur_performance = self._performance(tmp_edited_X, tmp_edited_y)
                    # if the performance start the decrease, disregard the delete and break the loop
                    if cur_performance < performance:
                        to_delete[i] = False
                        break
                    # the performance is getting better, update the performance to compare
                    else:
                        performance = cur_performance

        # Update X_train and y_train with edited version
        self.X_train = [self.X_train[i] for i in range(initial_size) if not to_delete[i]]
        self.y_train = [self.y_train[i] for i in range(initial_size) if not to_delete[i]]
        
        # print("Following rows in training set are deleted: ", [i for i in range(initial_size) if  to_delete[i]])
        

"""
Condensed K-Nearest Neighbor
When parameter gamma is used to initialize the Edited KNN
indicating this is a regression problem
Example usage:
knn_regressor = CondensedKNN(k=5, epsilon=0.1, gamma=0.1)  # for regression
knn_regressor.fit(X_train, y_train)
y_pred_regression = knn_regressor.predict(X_test)

knn_classifier = CondensedKNN(k=5, epsilon=0.1)  # for classification
knn_classifier.fit(X_train, y_train)
y_pred_classification = knn_classifier.predict(X_test)
"""
class CondensedKNN:
    def __init__(self, k=3, epsilon=0.1, gamma=None, cyclic={}):
        self.k = k
        self.epsilon = epsilon
        self.gamma = gamma
        self.cyclic = cyclic
    
    # store the training datasets
    # calculate the condensed datasets 
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self._condense_dataset()

    def predict(self, X_test):
        y_pred = []
        neighbors = []
        for x in X_test:
            neighbor_y, neighbor_x = get_neighbors(self.X_train, self.y_train, x, self.k, self.cyclic)
            neighbors.append([neighbor_y, neighbor_x])
            y_pred.append(determine_class(x, neighbor_y, neighbor_x, self.gamma, self.cyclic))
        return neighbors, y_pred

    def update_param(self, param_setting):
        self.k = param_setting['k']
        if 'epsilon' in param_setting.keys():
            self.epsilon = param_setting['epsilon']
            
    def _condense_dataset(self):
        # Add the first data point from the training set into the condensed set.
        condensed_X = []
        condensed_y = []
        condensed_X.append(self.X_train[0])
        condensed_y.append(self.y_train[0])
        
        # record the previous condensed_y to see when it does not change
        condensed_record = condensed_y
        
        while True:
            # For each data point, attempt to predict its value using the condensed set via 1-nn
            for i, x in enumerate(self.X_train):
                knn = KNN(k=1, gamma=self.gamma) 
                knn.fit(np.array(condensed_X), np.array(condensed_y))
                _, y_pred = knn.predict([x])

                # If the prediction is incorrect, add the data point to the condensed set.
                if not self._is_classified_correctly(y_pred, self.y_train[i]):
                    condensed_X.append(x)
                    condensed_y.append(self.y_train[i])
            # check if we have the same as previous condesed y
            if condensed_y == condensed_record:
                break
            else:
                condensed_record = condensed_y
                
        # update the training data with condensed 
        self.X_train = np.array(condensed_X)
        self.y_train = np.array(condensed_y)

    def _is_classified_correctly(self, pred, test):
        if self.gamma:
            # “correct” prediction if the prediction falls within some ε of the ground truth value
            if pred <= test + self.epsilon and pred >= test - self.epsilon:
                return True
        else:
            if pred == test:
                return True
        return False
