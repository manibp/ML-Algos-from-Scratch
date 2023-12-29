## Implementation of Decision Tree Algorithm from Scratch

## Design Template
# 1. Define base Node class with one parent and two child nodes
# 2. Write fit method
#   a. Iterate over all features and split values and calculate best split
#   b. compute best split using Entropy and Information gain
#   c. Recursively grow the tree to the left and right based on split thresholds
# 3. Write predict method
#   a. Traverse the tree based on the split thresholds and feature ids to the leaf node
#   b. Retrieve the value at the leaf node

import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, feature_idx = None, threshold =None, left=None, right=None,  value= None ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Classifier
class DecisionTreeClasssifier:
    def __init__(self, max_features = 10, max_depth =100, min_samples = 2 ):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features


    def fit(self, X,y):
        self.depth = 0
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self,X,y):
        n_samples, n_features  = X.shape
        n_classes = len(np.unique(y))

        ## Terminating condition:
        if n_samples < self.min_samples or self.depth >= self.max_depth or n_classes == 1:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)
        
        #find best split
        best_split_threshold, best_split_feature = self._best_split(X,y)

        left_ids, right_ids = self._split(X[:,best_split_feature], best_split_threshold)
        self.depth +=1

        left = self._grow_tree(X[left_ids,:], y[left_ids])
        right = self._grow_tree(X[right_ids,:], y[right_ids])
        return TreeNode(best_split_feature, best_split_threshold,left, right)
    
    def _best_split(self, X, y):
        _, no_features = X.shape
        feat_ids = np.random.choice(no_features, min(self.max_features, no_features), replace=False)
        best_gain = -1
        b_split_threshold = None
        b_split_feature = None
        
        # Iterate over each feature
        for feat_id in feat_ids:
            x =X[:,feat_id]
            split_values = np.unique(x)
            for split_thresh in split_values:
                left_ids, right_ids = self._split(x, split_thresh)
                gain = self._information_gain(y, left_ids, right_ids)
                if gain > best_gain:
                    best_gain =gain
                    b_split_feature = feat_id
                    b_split_threshold= split_thresh
        return b_split_threshold, b_split_feature
    
    def _entropy(self, y):
        pis =np.bincount(y)/len(y)
        return -np.sum([pi*np.log(pi) for pi in pis if pi >0])

    def _information_gain(self, y, left_ids, right_ids): 
        parent_entropy = self._entropy(y)
        
        left_child_entropy = self._entropy(y[left_ids])
        right_child_entropy = self._entropy(y[right_ids])

        n =len(y)
        n_l, n_r = len(left_ids), len(right_ids)
        return parent_entropy - (n_l*left_child_entropy +n_r*right_child_entropy)/n
    
    def _split(self, x, thresh):
        left_ids = np.where(x<= thresh)[0]
        right_ids = np.where(x > thresh)[0]
        return left_ids, right_ids

    def _most_common_label(self,y):
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
## Regressor
    
## Design Template
# 1. Define base Node class with one parent and two child nodes
# 2. Write fit method
#   a. Iterate over all features and split values and calculate best split
#   b. compute best split using Information gain as net decrease in standard deviation 
#   c. Recursively grow the tree to the left and right based on split thresholds
##  d. Use number of samples and coefficient of variation ((SD/mean)*100) as stopping criteria
# 3. Write predict method
#   a. Traverse the tree based on the split thresholds and feature ids to the leaf node
#   b. Retrieve the value at the leaf node and extract average value

class DecisionTreeRegressor:
    def __init__(self, max_features = 10, max_depth =100, min_samples = 3, min_coeff_deviation = 0.1 ):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        if not  (0 <= min_coeff_deviation <=1):
            raise ValueError("Coefficient of Deviation should be between 0 and 1 (inclusive)")
        self.min_coeff_deviation = min_coeff_deviation


    def fit(self, X,y):
        self.depth = 0
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self,X,y):
        n_samples, n_features  = X.shape
        n_classes = len(np.unique(y))
        CV = np.round((y.std()/(y.mean()+1e-9)*100))
        
        ## Terminating condition:
        if n_samples < self.min_samples or self.depth >= self.max_depth or CV < self.min_coeff_deviation*100:
            leaf_value = self._avg_label_value(y)
            return TreeNode(value=leaf_value)
        
        #find best split
        best_split_threshold, best_split_feature = self._best_split(X,y)

        left_ids, right_ids = self._split(X[:,best_split_feature], best_split_threshold)
        self.depth +=1

        left = self._grow_tree(X[left_ids,:], y[left_ids])
        right = self._grow_tree(X[right_ids,:], y[right_ids])
        return TreeNode(best_split_feature, best_split_threshold,left, right)
    
    def _best_split(self, X, y):
        _, no_features = X.shape
        feat_ids = np.random.choice(no_features, min(self.max_features, no_features), replace=False)
        best_gain = 0
        b_split_threshold = None
        b_split_feature = None
        
        # Iterate over each feature
        for feat_id in feat_ids:
            x =X[:,feat_id]
            split_values = np.unique(x)
            for split_thresh in split_values:
                left_ids, right_ids = self._split(x, split_thresh)
                gain = self._information_gain(y, left_ids, right_ids)
                if gain > best_gain:
                    best_gain =gain
                    b_split_feature = feat_id
                    b_split_threshold= split_thresh
        return b_split_threshold, b_split_feature
    
    def _entropy(self, y):
        return np.std(y) if len(y) >1 else 0

    def _information_gain(self, y, left_ids, right_ids): 
        parent_entropy = self._entropy(y)
        
        left_child_entropy = self._entropy(y[left_ids])
        right_child_entropy = self._entropy(y[right_ids])

        n =len(y)
        n_l, n_r = len(left_ids), len(right_ids)
        return parent_entropy - (n_l*left_child_entropy +n_r*right_child_entropy)/n
    
    def _split(self, x, thresh):
        left_ids = np.where(x<= thresh)[0]
        right_ids = np.where(x > thresh)[0]
        return left_ids, right_ids

    def _avg_label_value(self,y):
        return np.mean(y)
    
    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)