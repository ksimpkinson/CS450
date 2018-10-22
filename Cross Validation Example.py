from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict

# Toy data set modeled off of the Iris data set
data = [[3.0, 1.2, 4.5],
        [2.8, 2.0, 5.6],
        [1.4, 2.2, 5.2],
        [2.5, 1.5, 6.3],
        [3.1, 1.7, 5.7]]

# Toy target set modeled off of the Iris target set
target = [0, 0, 1, 2, 1]

# The number of splits I want
n = 10

# Get the KFolder, telling it the number of ways you want it to be split and that you want the data to be
# selected randomly.
kf = KFold(10, n_folds=n, shuffle=True)

# Store what your classifier returns in a list. There are alternate ways of doing this step
predictions = []

# I put in print statements just so that you could see what it was doing
for train_index, test_index in kf.split(data):
    print(train_index) # See the list. Be the list.
    print(test_index) # And do the same here
    print(data[train_index]) # Proof that it gets those indexes
    print(data[test_index]) # Notice that it collects different ones
    # collect all of your predictions. This is optional, depending on what else you are doing.
    predictions.append(classifier(data[train_index], # I don't know what your classifier does, but mine takes in all of these
                                  data[test_index], 
                                  target[train_index], # Notice that both target and data get indexed. This is important!
                                  target[test_index]))