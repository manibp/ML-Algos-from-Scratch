## Implementation of Random Forest Algorithm from Scratch

## Classifier

## Design Template
# 1. Import Decision Tree Classifier from Decision Trees module built from scratch
# 2. Write fit method
#   a. Build as many independent decision trees as defined by the user 
#   b. For each tree choose a random subset of features and sample data
#   c. Store the trees in a list for further use by predict method
# 3. Write predict method
#   a. Traverse each tree independently and obtain leaf node value
#   b. Aggregate them across all the trees (by Majority for classifier)

from DecisionTrees import DecisionTreeClasssifier, DecisionTreeRegressor
import numpy as np
from collections import Counter

class RandomForestCLassifier():
    def __init__(self, num_iterations = 100, prop_features = 0.7, prop_samples =0.7, max_depth =100, min_samples_split =2):
        self.num_iteration = num_iterations
        self.prop_features = prop_features
        self.prop_samples = prop_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        np.random.seed(1)

    def fit(self,X, y):
        self.forest =[self._grow_tree(X,y) for _ in range(self.num_iteration)]
        
    def _grow_tree(self, X,y):
        tot_s, tot_f = X.shape
        sub_f = np.round(tot_s*self.prop_features).astype(int)
        sub_s = np.round(tot_f*self.prop_samples).astype(int)

        # random selection of data samples
        sub_s_ids = np.random.randint(tot_s, size =sub_s)
        clf = DecisionTreeClasssifier(sub_f, self.max_depth, self.min_samples_split)
        clf.fit(X[sub_s_ids,:], y[sub_s_ids])
        return clf

    def predict(self, X):
        predictions = np.array([node.predict(X) for node in self.forest])
        preds_tr = predictions.T
        return np.array([self._most_common_label(pred) for pred in preds_tr])
    
    def _most_common_label(self,y):
        return Counter(y).most_common(1)[0][0]


# ## Regressor
    
## Design Template
# 1. Import Decision Tree Regressor from Decision Trees module built from scratch
# 2. Write fit method
#   a. Build as many independent decision trees as defined by the user 
#   b. For each tree choose a random subset of features and sample data
#   c. Store the trees in a list for further use by predict method
# 3. Write predict method
#   a. Traverse each tree independently and obtain leaf node value
#   b. Aggregate them across all the trees (by Majority for classifier)
    
class RandomForestRegressor():
    def __init__(self, num_iterations = 100, prop_features = 0.7, prop_samples =0.7, max_depth =100, min_samples_split =2, min_coeff_deviation =0.1):
        self.num_iteration = num_iterations
        self.prop_features = prop_features
        self.prop_samples = prop_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_coeff_deviation = min_coeff_deviation
        np.random.seed(1)

    def fit(self,X, y):
        self.forest =[self._grow_tree(X,y) for _ in range(self.num_iteration)]
        
    def _grow_tree(self, X,y):
        tot_s, tot_f = X.shape
        sub_f = np.round(tot_s*self.prop_features).astype(int)
        sub_s = np.round(tot_f*self.prop_samples).astype(int)

        # random selection of data samples
        sub_s_ids = np.random.randint(tot_s, size =sub_s)
        reg = DecisionTreeRegressor(sub_f, self.max_depth, self.min_samples_split, self.min_coeff_deviation)
        reg.fit(X[sub_s_ids,:], y[sub_s_ids])
        return reg

    def predict(self, X):
        predictions = np.array([node.predict(X) for node in self.forest])
        preds_tr = predictions.T
        return np.array([self._avg_label_value(pred) for pred in preds_tr])
    
    def _avg_label_value(self,y):
        return np.mean(y)
