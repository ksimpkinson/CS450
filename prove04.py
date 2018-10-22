"""
Decision Tree Milestone
October 13, 2018
"""

# Load libraries
from sklearn import tree
from sklearn.datasets import load_iris
# import graphviz
import pydotplus

# Load data
iris = load_iris()

# Assign attributes and targets
X = iris.data
Y = iris.target

# Create decision tree classifier object
classifier = tree.DecisionTreeClassifier(random_state=0)

# Train model
model = classifier.fit(X, Y)

# Create DOT data (for visual)
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Create pdf
graph.write_pdf("iris.pdf")

# Create png
graph.write_png("iris.png")


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

