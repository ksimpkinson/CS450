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


def load_iris_data():
    """
    Load iris dataset
    """
    iris = load_iris()
    
    feature_names = iris.feature_names
    
    target_names = iris.target_names

    # Assign attributes and targets
    X = iris.data
    Y = iris.target
    
    return X, Y, feature_names, target_names

# Load lenses data



# Load credit data



def create_train_test(X, Y, features, targets):
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
    model = classifier.fit(x_train, y_train)
    
    # Visuals

    # Create DOT data (for visual)
    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=features,
                                    class_names=targets,
                                    filled=True, rounded=True,
                                    special_characters=True)

    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Create png
    file_name = input("Enter file name: ")
    graph.write_png(file_name)

    # Predict
    predictions = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)*100
    
    return accuracy


def bin_data(X):
    """
    Create bins for all numeric data
    """
    # Create bins
    bins = np.array([0, 2, 4, 6, 8])

    # Bin training data
    x_binned = np.digitize(X, bins)
    
    return x_binned


# Missing Data

# Remove missing values



"""
Other data to use:
Lenses -
    A1. Age - 1: Young; 2: Pre-presbyopic; 3: Presbyopic (Far sighted)
    A2. Prescription - 1: Myope (near-sighted); 2: Hypermetrope (far-sighted)
    A3. Astigmatic (blurred vision) - 1: No; 2: Yes
    A4. Tear Production - 1: Reduced; 2: Normal
    Class - 1: Hard contact lenses; 2: Soft contact lenses; 3: No lenses
Credit -
    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    Class: +,-         (class attribute) 
"""


def main():
    iris_x, iris_y, iris_features, iris_targets = load_iris_data()
    
    normal_accuracy = create_train_test(iris_x, iris_y, iris_features, iris_targets)
    
    print("Normal accuracy:\n{0:.2f}%".format(normal_accuracy))
    
    iris_x_binned = bin_data(iris_x)
    
    print("Binned accuracy:\n{0:.2f}%".format(create_train_test(iris_x_binned, iris_y,
                                                                iris_features, iris_targets)))
    
    

if __name__ == "__main__":
    main()
    
    
# One Hot
"""
Label encode data
One hot encode data
"""
# One Hot Strategy
iris = load_iris()

X = iris.data
Y = iris.target

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

X[:, 1] = label_encoder.fit_transform(X[:, 1])
X[:, 2] = label_encoder.fit_transform(X[:, 2])
X[:, 3] = label_encoder.fit_transform(X[:, 3])

x_onehot = onehot_encoder.fit_transform(X)
    
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 15)

# Create decision tree classifier object
classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=4)

# Train model
model = classifier.fit(x_train, y_train)
    
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
file_name = input("Enter file name: ")
graph.write_png(file_name)

# Predict
predictions = classifier.predict(x_test)

accuracy = accuracy_score(y_test, predictions)*100

print("One hot accuracy:\n{0:.2f}%".format(accuracy))

