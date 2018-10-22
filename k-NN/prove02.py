from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import collections

# Load the data
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
print("Targets:", iris.target)

# Show the actual target names that correspond to each number
print("Target names:", iris.target_names)

# Create a training set
iris_train, iris_test, target_train, target_test = train_test_split(iris.data, iris.target,
                                                    test_size = 0.3, random_state = 15)

# View training set and target set
#print("Training data:\n", iris_train)
print("Target data:\n", target_test)

# Scale features
scaler = StandardScaler()  # Assign a shorter name, easier to reference
scaler.fit(iris_train)     # Fit the scaler to the training data

# Print test data before it's scaled
print(iris_test)

# Basically convert all numbers to a z-score
iris_train = scaler.transform(iris_train)  # Rescale each attribute in the train data
iris_test = scaler.transform(iris_test)    # Reslace each attribute in the test data

# Print test data after it's been scaled to compare
print(iris_test)

# View the premade algorithm
classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(iris_train, target_train)
predictions = model.predict(iris_test)

# View the predictions and accuracy
print("Predicted targets for pre-made algorithm:\n", predictions)
print("Actual targets:\n", target_test)
print("Accuracy of pre-made algorithm:\n{0:.2f}%".format(accuracy_score(target_test, predictions)*100, "%"))

# My algorithm
# I referenced (https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)
# to learn how to code my own algorithm
# Create new class of hardcoded algorithm
# Find distance between test point and data train points
# (x_test1 - x_train1)^2 + (x_test2 - x_train2)^2

# Normally a training algorithm would be needed before prediction
# The k-NN process doesn't require training
def train(iris_train, target_train):
    # do nothing
    return

def predict(iris_train, target_train, iris_test, k):
    # create empty lists for distances and targets
    # to be filled as we loop through the rows and calc distances
    distances = []
    targets = []
    
    for i in range(len(iris_train)):
        # [(x1 - x'1)^2 + (x2 - x'2)^2 + (x3 - x'3)^2]
        # Calculate the distances from the test data to each point
        # in the training data
        # [i, :] means grab the ith row and all the columns
        distance = np.sum(np.square(iris_test - iris_train[i, :]))
        # Add those distances to the distance list
        distances.append([distance, i])
        
    # Sort through the list of distances, ordering smallest to largest
    distances = sorted(distances)
    
    # Grab all k neighbors to the test data
    for i in range(k):                       # Loop through k times
        index = distances[i][1]              # Set index to be the ith distance
        targets.append(target_train[index])  # Add that distance to the targets list
        
    # Return the single most common/most repeated target
    return collections.Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(x_train, y_train, x_test, predictions, k):
    # Train on the input data
    # Normally we would need this
    # For k-NN this is not needed
    train(x_train, y_train)

    # Loop through all observations in the test data
    for i in range(len(x_test)):
        # Add a prediction to the prediction list
        # Pass the training data, training target, and the current row from the test data
        # Only need the current row of the test data to be able to compare that row
        predictions.append(predict(x_train, y_train, x_test[i, :], k))

 
def main():
    predictions = []
    
    kNearestNeighbor(iris_train, target_train, iris_test, predictions, 8)
    
    # Convert the list to be an array
    # Needs to be a numpy array to work properly w/ accuracy_score
    predictions = np.asarray(predictions)
    
    # Print accuracy
    accuracy = accuracy_score(target_test, predictions)
    print("My k-NN accuracy: {0:.2f}%".format(accuracy*100))
 
 
if __name__ == "__main__":
    main()
    
