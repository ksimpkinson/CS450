"""
Decision Tree Milestone
October 13, 2018
"""

# Load libraries
from sklearn import tree
from sklearn.datasets import load_iris
import pydotplus
import graphviz
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


iris = load_iris()
    
X = iris.data
Y = iris.target

"""
Create training and testing data
Create classifier and model
Create predictions and return accuracy
"""
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 15)

# Create decision tree classifier object
classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=4)

# Train model
classifier = classifier.fit(x_train, y_train)
    
# Visuals

# Create DOT data (for visual)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Create png
graph.write_png("iris.png")

# Predict
predictions = classifier.predict(x_test)

print("Normal predictions:\n", predictions)

print("Actual classes:\n", y_test)

accuracy = accuracy_score(y_test, predictions)*100
    
print("Normal accuracy:\n{0:.2f}%".format(accuracy))
    

"""
Create bins for all numeric data
"""
# Create bins
bins = np.array([0, 2, 4, 6, 8])

# Bin training data
x_binned = np.digitize(X, bins)

x_train, x_test, y_train, y_test = train_test_split(x_binned, Y, test_size = 0.3,
                                                    random_state = 15)

classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=4)

classifier = classifier.fit(x_train, y_train)

# Create DOT data (for visual)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Create png
graph.write_png("iris_binned.png")

# Predict
predictions = classifier.predict(x_test)

print("Bin predictions:\n", predictions)

print("Actual classes:\n", y_test)

accuracy = accuracy_score(y_test, predictions)*100

print("Binned accuracy:\n{0:.2f}%".format(accuracy))
    

# One Hot
"""
Label encode data
One hot encode data
"""
# One Hot Strategy
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

x_onehot = X
y_onehot = Y

x_onehot[:, 0] = label_encoder.fit_transform(x_onehot[:, 0])
x_onehot[:, 1] = label_encoder.fit_transform(x_onehot[:, 1])
x_onehot[:, 2] = label_encoder.fit_transform(x_onehot[:, 2])

x_onehot = onehot_encoder.fit_transform(x_onehot)
    
x_train, x_test, y_train, y_test = train_test_split(x_onehot, Y, test_size = 0.3,
                                                    random_state = 15)

classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=4)

# Train model
classifier = classifier.fit(x_train, y_train)

# Create DOT data (for visual)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
    
# Create png
graph.write_png("iris_onehot.png")

# Predict
predictions = classifier.predict(x_test)

print("One hot predictions:\n", predictions)

print("Actual classes:\n", y_test)

accuracy = accuracy_score(y_test, predictions)*100

print("One hot accuracy:\n{0:.2f}%".format(accuracy))



# Lenses

features = ["age", "pres", "blurry", "tear"]
classes = ["hard lenses", "soft lenses", "no lenses"]

headers = ["age", "pres", "blurry", "tear", "lense"]

lenses = pd.read_csv("lenses.csv",
                     header=None, names=headers)

print("\nLenses data:\n", lenses.head())

lenses_np = lenses.values

X = lenses_np[:, 0:-1]
Y = lenses_np[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 15)

# Create decision tree classifier object
classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=4)

# Train model
classifier = classifier.fit(x_train, y_train)
    
# Visuals

# Create DOT data (for visual)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=features,
                                class_names=classes,
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Create png
graph.write_png("lense.png")

# Predict
predictions = classifier.predict(x_test)

print("Lense predictions:\n", predictions)

print("Actual classes:\n", y_test)

accuracy = accuracy_score(y_test, predictions)*100
    
print("Lense accuracy:\n{0:.2f}%".format(accuracy))




# Credit

features = ["A1", "A2", "A3", "A4", "A5",
            "A6", "A7", "A8", "A9", "A10",
            "A11", "A12", "A13", "A14", "A15"]
classes = ["+", "-"]

credit = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
                     header=None)

print(credit.head())

credit = credit.replace("?", np.nan)

credit = credit.dropna()

credit_np = credit.values

X = credit_np[:, 0:-1]
Y = credit_np[:, -1]

X[:, 0] = label_encoder.fit_transform(X[:, 0])
X[:, 3] = label_encoder.fit_transform(X[:, 3])
X[:, 4] = label_encoder.fit_transform(X[:, 4])
X[:, 5] = label_encoder.fit_transform(X[:, 5])
X[:, 6] = label_encoder.fit_transform(X[:, 6])
X[:, 8] = label_encoder.fit_transform(X[:, 8])
X[:, 9] = label_encoder.fit_transform(X[:, 9])
X[:, 11] = label_encoder.fit_transform(X[:, 11])
X[:, 12] = label_encoder.fit_transform(X[:, 12])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 15)

# Create decision tree classifier object
classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=4)

# Train model
classifier = classifier.fit(x_train, y_train)
    
# Visuals

# Create DOT data (for visual)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=features,
                                class_names=classes,
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Create png
graph.write_png("credit.png")

# Predict
predictions = classifier.predict(x_test)

print("Credit predictions:\n", predictions)

print("Actual classes:\n", y_test)

accuracy = accuracy_score(y_test, predictions)*100
    
print("Credit accuracy:\n{0:.2f}%".format(accuracy))





