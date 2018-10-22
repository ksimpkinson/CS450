from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
import pydotplus
import collections

X = [[0, 0], [1, 1]]
Y = [0, 1]

# Takes two parameters
# X w/ size (n_samples, n_features) - Attributes
# Y w/ size (n_samples) - Targets
clf = tree.DecisionTreeClassifier()

# Send both arrays through fitting model
# What does the 'fit' function do?
# Training on the data?
clf = clf.fit(X, Y)

# Once the model has been fit, it can be predicted
print(clf.predict([[2., 2.]]))



##### Iris dataset

iris = load_iris()
clf = clf.fit(iris.data, iris.target)

# Using graphviz we can export the tree
dot_graph = tree.export_graphviz(clf, out_file = None)

graph = graphviz.Source(dot_graph)

   # Other features to color the graph
dot_data = tree.export_graphviz(clf, out_file = 'tree.dot',
                                 feature_names = iris.feature_names,
                                 class_names = iris.target_names,
                                 filled = True, rounded = True,
                                 special_characters = True)
import pydot

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

#graph = graphviz.Source(dot_data)
#graph



################# Another Example #########################

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
import pydot

# Data Collection
X = [[180, 15, 0],
     [177, 42, 0],
     [136, 35, 1],
     [174, 65, 0],
     [141, 28, 1]]

Y = ['man', 'woman', 'woman', 'man', 'woman']

data_feature_names = ['height', 'hair length', 'voice pitch']

# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
#graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph = pydotplus.graph_from_dot_data(dot_data)

#graph.draw('tree.png')

graph.write_png('tree2.png')






############### One more try #########################

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
import pydotplus

# Load data
iris = datasets.load_iris()

X = iris.data
Y = iris.target

# Create decision tree classifier object
clf = DecisionTreeClassifier(random_state = 0)

# Train model
model = clf.fit(X, Y)






# Visuals

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file = None,
                               feature_names = iris.feature_names,
                               class_names = iris.target_names)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())

# Create pdf
graph.write_pdf("iris.pdf")

# Create png
graph.write_png("iris.png")





