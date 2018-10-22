from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

# Create a training set
iris_train, iris_test, target_train, target_test = train_test_split(iris.data, iris.target,
                                                    test_size = 0.3, random_state = 0)

# View training set and target set
print("Training data:\n", iris_train, "\n")
print("Target data:\n", target_test, "\n")
print("Target train:\n", target_train, "\n")

# Premade algorithm
classifier = GaussianNB()
model = classifier.fit(iris_train, target_train)

# Make predictions
targets_predicted = model.predict(iris_test)

print("Predicted targets:\n", targets_predicted)
print("Actual targets:\n", target_test)

print("Accuracy:\n", accuracy_score(target_test, targets_predicted)*100, "%")

# Create new class
class HardCodedClassifier:
    def fit(self, data_train, targets_train):
        return ""
    
    def predict(self, data_test):
        prediction = np.array([])
        for i in data_test:
            prediction = np.append(prediction, 0)
        return prediction           
    

classifier = HardCodedClassifier()
model = classifier.fit(iris_train, target_train)
targets_predicted = classifier.predict(iris_test)

print(targets_predicted)
print(target_test)

print("Accuracy:\n{0:.2f}%".format(accuracy_score(target_test, targets_predicted)*100))
 
 
    