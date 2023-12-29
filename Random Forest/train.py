from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForestCLassifier, RandomForestRegressor

# Classification Task
# data = datasets.load_breast_cancer()
# X, y = data.data, data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf = RandomForestCLassifier(prop_samples=0.6)
# clf.fit(X_train, y_train)

# predictions = clf.predict(X_test)

# def accuracy(y_test, y_pred):
#     return np.sum(y_test ==y_pred)/len(y_test)
# acc = accuracy(y_test, predictions)
# print(acc)


# Regression Task
data = datasets.load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(max_depth=10)
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

def Rsquared(y_test, y_pred):
    return 1 - np.sum(np.square(y_pred-y_test))/(np.sum(np.square(y_test-y_test.mean())))
    
r2 = Rsquared(y_test, predictions)
print(r2)