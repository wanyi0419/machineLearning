import numpy as np
import pandas as pd
from math import log2
from collections import Counter
import random

'''
Demonstration Function
Functions to demonstrate the calculations embeded in the class.
The class does not call these functions,
but the implementation is same as the functions used in the class
'''
# to be used to calculate the information gain
def calculate_entropy(y):
        ''' function to calculate the entropy of y'''
        entropy = 0
        total_count = len(y)
        for label in set(y):
            label_count = np.sum(y == label)
            p = label_count / total_count
            entropy -= p * log2(p)
        return entropy

def calculate_information_gain(X, y, feature_idx):
    ''' function that calculate the information gain of given X and y'''
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    weighted_entropy = 0

    for value, count in zip(values, counts):
        subset_indices = X[:, feature_idx] == value
        subset_entropy = calculate_entropy(y[subset_indices])
        weighted_entropy = weighted_entropy + (count / len(X)) * subset_entropy

    return total_entropy - weighted_entropy

def calculate_gain_ratio(X, y, feature_idx):
    ''' function to calculate the gain ratio '''
    values, counts = np.unique(X[:, feature_idx], return_counts=True)

    gain = calculate_information_gain(X, y, feature_idx)

    split_info = -sum([(count / len(X)) * log2(count / len(X)) for count in counts])
    if split_info == 0:
        split_info = 1  # Avoid division by zero

    # gain ration is gain/IV
    gain_ratio = gain / split_info

    return gain_ratio

def calculate_mse(left_y, right_y):
    ''' calculate mse, noting the calculation of the leaf node
        for decision tree regressor is the mean of y in the partition'''
    left_mse = 0
    right_mse = 0

    # when there is no elements in the list, mse=0
    if len(left_y) != 0:
        # y_hat is calculated by the mean of y in the partition
        left_y_hat = np.mean(left_y)
        left_mse = np.mean((left_y - left_y_hat) ** 2)

    # when there is no elements in the list, mse=0
    if len(right_y) != 0:
        # y_hat is calculated by the mean of y in the partition
        right_y_hat = np.mean(right_y)
        right_mse = np.mean((right_y - right_y_hat) ** 2)

    # total mse = left mse + right mse 
    return left_mse + right_mse

def whether_to_prune(X, y, tree):
    '''
    decide whether the subtree should be pruned (regression tree)
    - if the tree should be pruned, print yes
    - the function is not implemented explicity in the classes
      it is embeded in the _reduced_error_prune_tree method
    '''
    # Prune the subtree if it reduces error rate (evaluated by mse)
    error_before_prune = calculate_error(X, y, tree)
    # the pruned sub tree is replaced by the mean value in the current partition of y
    error_after_prune = calculate_error(X, y, np.mean(y))
    if error_after_prune <= error_before_prune:
        print("yes")

def calculate_error(X, y, node):
    ''' function to calculate prediction error (mse) for regressor'''

    y_pred = [predict_tree(x, node) for x in X]
    mse = np.mean((y_pred - y) ** 2)
    return mse

def predict_tree(x, node):
    '''function to predict for a given data x'''
    # leaf node is denoted by a single value
    # move to the child node if the node in form of dictionary (intermediate node)
    if isinstance(node, dict):
        feature_idx = node["feature_index"]
        threshold = node["threshold"]
        # find with side are we moving
        if x[feature_idx] <= threshold:
            return predict_tree(x, node["left"])
        else:
            return predict_tree(x, node["right"])
    else:
        return node

'''
Decision Tree Classifier
Based on ID3 method
X: array of features, numerical features should be preprocessed before passing in using the binary methods
y: array of corresponding targets
decision criteria: gain ratio
prune tree option: reduced error prune, could be call after fit using tree.reduced_error_prune_tree(X,y)
'''
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        ''' constructor '''
        # The generation of the tree stops when max depth is met
        self.max_depth = max_depth
        # minimum samples for each split
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        ''' function to fit the data to a tree '''
        self.n_features = X.shape[1]
        # build the tree
        self.tree = self._build_tree(X, y, depth=0)

    def reduced_error_prune_tree(self, X, y):
        ''' function to prune a tree given a dataset'''      
        self.tree = self._reduced_error_prune_tree(X, y, self.tree)

    def _reduced_error_prune_tree(self, X, y, node):
        ''' function to prune a tree given a dataset'''
        ''' a sub tree is pruned if pruning it could reduce the error'''

        # We only prune the tree when the node is an intermediate node
        if isinstance(node, dict) and len(X)>0 and len(y)>0:
            for value, child_node in node["children"].items():
                # compute the X and y with current feature == the value
                feature_idx = node["feature_index"]
                sub_indices = X[:, feature_idx] == value
                sub_X, sub_y= X[sub_indices], y[sub_indices] 

                # Prune the subtree for each child with the corresponding sub X and y
                node["children"][value] = self._reduced_error_prune_tree(sub_X, sub_y, child_node)

            # calculate the error rate with and without the subtree
            error_rate_before_prune = self._calculate_error_rate(X, y, node)
            # the pruned sub tree is replaced by the most common value in y
            error_rate_after_prune = self._calculate_error_rate(X, y, Counter(y).most_common(1)[0][0])
            # Prune the subtree if it reduces error rate
            if error_rate_after_prune <= error_rate_before_prune:
                return Counter(y).most_common(1)[0][0]

        return node

    def _calculate_error_rate(self, X, y, node):
        ''' function to calculate error rate for classifier'''
        # predict y for each x in X
        y_pred = [self._predict_tree(x, node) for x in X]
        error_count = np.sum(y != y_pred)
        error_rate = error_count / len(y)
        return error_rate

    def _calculate_entropy(self, y):
        ''' function to calculate the entropy of y'''
        entropy = 0
        total_count = len(y)
        for label in set(y):
            label_count = np.sum(y == label)
            p = label_count / total_count
            entropy -= p * log2(p)
        return entropy

    def _calculate_gain_ratio(self, X, y, feature_idx):
        ''' function to calculate the gain ratio '''
        ''' gain is calculated by total entropy - weighted entropy'''
        total_entropy = self._calculate_entropy(y)
        values, counts = np.unique(X[:, feature_idx], return_counts=True)
        weighted_entropy = 0

        for value, count in zip(values, counts):
            subset_indices = X[:, feature_idx] == value
            subset_entropy = self._calculate_entropy(y[subset_indices])
            weighted_entropy = weighted_entropy + (count / len(X)) * subset_entropy

        split_info = -sum([(count / len(X)) * log2(count / len(X)) for count in counts])
        if split_info == 0:
            split_info = 1  # Avoid division by zero

        gain = total_entropy - weighted_entropy
        # gain ration is gain/IV
        gain_ratio = gain / split_info

        return gain_ratio

    def _find_best_split(self, X, y):
        ''' function to find the best split feature index'''
        best_gain_ratio = -1
        best_feature_idx = None

        # for each feature, evaluate the gain ratio
        for feature_idx in range(self.n_features):
            gain_ratio = self._calculate_gain_ratio(X, y, feature_idx)
            # the gain ratio is better, update the bested gain ration and feature index
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature_idx = feature_idx

        return best_feature_idx

    def _build_tree(self, X, y, depth):
        ''' recursive function to build the tree '''
        # split until stopping conditions are met and return the leaf node
        if len(set(y)) == 1 or depth == self.max_depth or len(X) < self.min_samples_split:
            # leaf node is denoted by a single value of class
            # using the most common y in the current partition
            return Counter(y).most_common(1)[0][0]

        best_feature_idx = self._find_best_split(X, y)
        # if cannot find best split, stop splitting and return the leaf node
        if best_feature_idx is None:
            return Counter(y).most_common(1)[0][0]

        node = {"feature_index": best_feature_idx, "children": {}}
        values, counts = np.unique(X[:, best_feature_idx], return_counts=True)

        for value, count in zip(values, counts):
            subset_indices = X[:, best_feature_idx] == value
            sub_X, sub_y = X[subset_indices], y[subset_indices]
            # try to build a tree for each child node
            node["children"][value] = self._build_tree(sub_X, sub_y, depth + 1)

        return node

    def predict(self, X):
        ''' function to predict the new dataset '''
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        ''' function to predict a single data point '''
        # leaf node is denoted by a single value
        # move to the child node if the node in form of dictionary (intermediate node)
        if isinstance(node, dict):
            feature_idx = node["feature_index"]
            value = x[feature_idx]
            if value in node["children"]:
                return self._predict_tree(x, node["children"][value])
            else: # if there is no such occurance, assign a randome choice
                available_children = list(node["children"].keys())
                random_choice = random.choice(available_children)
                return self._predict_tree(x, node["children"][random_choice])
        return node



'''
Decision Tree Regressor
Based on CART method
X: array of features, numerical features should be preprocessed before passing in using the binary methods
y: array of corresponding targets
decision criteria: mean square error
prune tree option: reduced error prune, could be call after fit using tree.reduced_error_prune_tree(X,y)
'''
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        ''' constructor '''
        # The generation of the tree stops when max depth is met
        self.max_depth = max_depth
        # minimum samples for each split
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        '''function to fit X and y'''
        self.n_features = X.shape[1]
        # build the tree
        self.tree = self._build_tree(X, y, depth=0)

    def reduced_error_prune_tree(self, X, y):
        ''' function to prune the existing tree using reduced error pruning'''
        self.tree = self._reduced_error_prune_tree(X, y, self.tree)

    def _reduced_error_prune_tree(self, X, y, node):
        ''' function for reduced error pruning'''
        # We only prune the tree when the node is an intermediate node
        # intermediate node is denoted by dict type
        # leaf node contains only a single value
        if isinstance(node, dict):

            # split the current data into left and right sub trees
            feature_idx = node["feature_index"]
            threshold = node["threshold"]
            left_indices = X[:, feature_idx] <= threshold
            right_indices = X[:, feature_idx] > threshold

            left_X, left_y = X[left_indices], y[left_indices]
            right_X, right_y = X[right_indices], y[right_indices]

            # attempt to prune the left and right sub trees
            node["left"] = self._reduced_error_prune_tree(left_X, left_y, node["left"])
            node["right"] = self._reduced_error_prune_tree(right_X, right_y, node["right"])

            # Prune the subtree if it reduces error rate (evaluated by mse)
            error_before_prune = self._calculate_error(X, y, node)
            # the pruned sub tree is replaced by the mean value in the current partition of y
            error_after_prune = self._calculate_error(X, y, np.mean(y))
            if error_after_prune <= error_before_prune:
                return np.mean(y)

        return node

    def _calculate_error(self, X, y, node):
        ''' function to calculate prediction error (mse) for regressor'''

        y_pred = [self._predict_tree(x, node) for x in X]
        mse = np.mean((y_pred - y) ** 2)
        return mse

    def _calculate_mse(self, left_y, right_y):
        ''' calculate mse, noting the calculation of the leaf node is the mean of y in the partition'''

        left_mse = 0
        right_mse = 0

        # when there is no elements in the list, mse=0
        if len(left_y) != 0:
            # y_hat is calculated by the mean of y in the partition
            left_y_hat = np.mean(left_y)
            left_mse = np.mean((left_y - left_y_hat) ** 2)

        # when there is no elements in the list, mse=0
        if len(right_y) != 0:
            # y_hat is calculated by the mean of y in the partition
            right_y_hat = np.mean(right_y)
            right_mse = np.mean((right_y - right_y_hat) ** 2)

        return left_mse + right_mse

    def _find_best_split(self, X, y):
        ''' find the best split based on minimizing mse'''

        best_mse = float("inf")
        best_feature_idx = None
        best_threshold = None

        # for each feature, each threshold, find the combination produces the best mse
        for feature_idx in range(self.n_features):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                left_y, right_y = y[left_indices], y[right_indices]

                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue
                # final mse is calculated by the left mse + right mse
                total_mse = self._calculate_mse(left_y, right_y)

                # if mse is better, update the best params
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_mse

    def _build_tree(self, X, y, depth):
        '''build a tree recursively'''

        # stop when max_depth is reached or the minimum sample split is reached
        # return leaf node: calculated as the mean of y in the current partition
        if len(X) == 0 or depth == self.max_depth or len(X) < self.min_samples_split:
            return np.mean(y)

        best_feature_idx, best_threshold, best_mse = self._find_best_split(X, y)

        # there is no best feature
        # return leaf node: calculated as the mean of y in the current partition
        if best_feature_idx is None:
            return np.mean(y)

        left_indices = X[:, best_feature_idx] <= best_threshold
        right_indices = ~left_indices

        # attempt the generate left and right sub trees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # return intermediate node
        return {
            "feature_index": best_feature_idx,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
            "mse": best_mse
        }

    def predict(self, X):
        '''function to predict for a given dataset X'''
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        '''function to predict for a given data x'''
        # leaf node is denoted by a single value
        # move to the child node if the node in form of dictionary (intermediate node)
        if isinstance(node, dict):
            feature_idx = node["feature_index"]
            threshold = node["threshold"]
            # find which side are we moving
            if x[feature_idx] <= threshold:
                return self._predict_tree(x, node["left"])
            else:
                return self._predict_tree(x, node["right"])
        else:
            return node