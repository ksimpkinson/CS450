# Hyperparameters:
# learning rate
# number of iterations
# actication functions

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def prep_glass(data_path):
    header = ["ID", "Refractive Index", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass Type"]

    glass = pd.read_csv(data_path,
                        header=None, names=header)

    print(glass.head())

    glass = glass.values

    X = glass[:, 0:-1]
    Y = glass[:, -1]

    # Normalize data
    X_norm = normalize(X, norm="l1")
    
    return X_norm, Y


def prep_census(data_path):
    
    header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
              "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
              "salary"]
    
    census = pd.read_csv(data_path,
                         header=None, names=header)

    print(census.head())
    
    census = census.replace("?", np.nan)
    
    census = census.dropna()
    
    census = census.values
    
    X = census[:, 0:-1]
    Y = census[:, -1]
     
    X[:, 1] = label_encoder.fit_transform(X[:, 1])
    X[:, 3] = label_encoder.fit_transform(X[:, 3])
    X[:, 5] = label_encoder.fit_transform(X[:, 5])
    X[:, 6] = label_encoder.fit_transform(X[:, 6])
    X[:, 7] = label_encoder.fit_transform(X[:, 7])
    X[:, 8] = label_encoder.fit_transform(X[:, 8])
    X[:, 9] = label_encoder.fit_transform(X[:, 9])
    X[:, 13] = label_encoder.fit_transform(X[:, 13])
    
    X = normalize(X)
    
    return X, Y


def prep_screening(data_path):
    
    header = ["Question 1",        "Question 2",        "Question 3",
              "Question 4",        "Question 5",        "Question 6",
              "Question 7",        "Question 8",        "Question 9",
              "Question 10",       "Age",               "Gender",
              "Ethnicity",         "Born w/ Jaundice",  "Fam Mem w/ PDD",  
              "Country of Res",    "Used Before",       "Screening Score",
              "Age Description",   "Relation",          "Class/ASD"]

    autism_screen = pd.read_csv(data_path, header=None, names=header)

    print(autism_screen.head())
    
    autism_screen = autism_screen.replace("?", np.nan)
    
    autism_screen = autism_screen.dropna()
    
    autism_screen = autism_screen.values
    
    X = autism_screen[:, 0:-1]
    Y = autism_screen[:, -1]
    
    X[:, 10] = label_encoder.fit_transform(X[:, 10])
    X[:, 11] = label_encoder.fit_transform(X[:, 11])
    X[:, 12] = label_encoder.fit_transform(X[:, 12])
    X[:, 13] = label_encoder.fit_transform(X[:, 13])
    X[:, 14] = label_encoder.fit_transform(X[:, 14])
    X[:, 15] = label_encoder.fit_transform(X[:, 15])
    X[:, 16] = label_encoder.fit_transform(X[:, 16])
    X[:, 18] = label_encoder.fit_transform(X[:, 18])
    X[:, 19] = label_encoder.fit_transform(X[:, 19])
    
    X = normalize(X)
    
    return X, Y


def train_test(X, Y):
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 7)
    
    classifier = MLPClassifier(max_iter=300, random_state=15, learning_rate_init=0.001,
                               activation="logistic")

    model = classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    accuracy = classifier.score(x_test, y_test)
    
    return accuracy


def main():
    
    glass_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

    glass_x, glass_y = prep_glass(glass_path)
    
    g_accuracy = train_test(glass_x, glass_y)
    
    print("Accuracy:\n{0:.2f}%".format(g_accuracy*100))
    
    census_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    census_x, census_y = prep_census(census_path)
    
    c_accuracy = train_test(census_x, census_y)
    
    print("Accuracy:\n{0:.2f}%".format(c_accuracy*100))
    
    autism_path = "Autism-Adult-Data.csv"
    
    autism_x, autism_y = prep_screening(autism_path)
    
    a_accuracy = train_test(autism_x, autism_y)
    
    print("Accuracy:\n{0:.2f}%".format(a_accuracy*100))
    
    
if __name__ == "__main__":
    main()

