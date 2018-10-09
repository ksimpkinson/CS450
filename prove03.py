"""
Prove Assignment Week 3
Oct 6, 2018

Requirements:

Read data from text files (e.g. comma- or space-delimited)
Handle non-numeric data
Handle missing data
Use k-fold Cross Validation
(Can use off the shelf implementation of kNN)
"""
# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
import collections

label_encoder = LabelEncoder()


def load_car_data(data_path):
    """
    No missing values
    """
    headers = ["buying",   "maint",
               "doors",    "persons",
               "lug_boot", "safety",
               "class"]
    
    df_car = pd.read_csv(data_path,
                         header = None,
                         names = headers)
    
    print("Cars Types Before:\n", df_car.dtypes, "\n")
    
    df_car_np = df_car.values
    
    car_data = df_car_np[:, 0:5]
    
    car_targets = df_car_np[:, 6]
    
    for col in range(car_data.shape[1]):
        car_data[:, col] = label_encoder.fit_transform(car_data[:, col])
        
    print("Car Data:\n", car_data, "\n")
    print("Car Targets:\n", car_targets, "\n")
    
    return car_data, car_targets


def load_autism_data(data_path):
    """
    Missing values are "?"
    95 rows are missing values, remove them
    Another option is to fill them with most frequent value
    """
    headers = ["Question 1",        "Question 2",        "Question 3",
               "Question 4",        "Question 5",        "Question 6",
               "Question 7",        "Question 8",        "Question 9",
               "Question 10",       "Age",               "Gender",
               "Ethnicity",         "Born w/ Jaundice",  "Fam Mem w/ PDD",  
               "Country of Res",    "Used Before",       "Screening Score",
               "Age Description",   "Relation",          "Class/ASD"]
    
    df_autism = pd.read_csv(data_path,
                            header = None,
                            names = headers)
    
    print("Autism Types Before:\n", df_autism.dtypes, "\n")
    
    df_autism = df_autism.replace("?", np.nan)
    
    # In columns 12 and 19 replace NaN values with most popular value
    
    print("Dim Before:\n", df_autism.shape, "\n")
    
    # Remove 95 rows that are missing values
    df_autism = df_autism.dropna()
    
    print("Dim After:\n", df_autism.shape, "\n")
    
    df_autism_np = df_autism.values
    
    autism_data = df_autism_np[:, 0:-1]
    autism_targets = df_autism_np[:, -1]
    
    #for col in range(autism_objects):
    #    if dtype == object:
    autism_data[:, 10] = label_encoder.fit_transform(autism_data[:, 10])
    autism_data[:, 11] = label_encoder.fit_transform(autism_data[:, 11])
    autism_data[:, 12] = label_encoder.fit_transform(autism_data[:, 12])
    autism_data[:, 13] = label_encoder.fit_transform(autism_data[:, 13])
    autism_data[:, 14] = label_encoder.fit_transform(autism_data[:, 14])
    autism_data[:, 15] = label_encoder.fit_transform(autism_data[:, 15])
    autism_data[:, 16] = label_encoder.fit_transform(autism_data[:, 16])
    autism_data[:, 18] = label_encoder.fit_transform(autism_data[:, 18])
    autism_data[:, 19] = label_encoder.fit_transform(autism_data[:, 19])
    
    print("Autism After:\n", autism_data, "\n")
    
    return autism_data, autism_targets
    
    
# Load mpg data
# Missing values are "?"
# Only missing 6 rows, remove them
def load_mpg_data(data_path):
    """
    Missing values are "?"
    Only 6 rows are missing, remove them
    """
    headers = ["mpg",         "cylinders", "displacement",
               "horsepower",  "weight",    "acceleration",
               "model year",  "origin",    "car name"]
    
    df_mpg = pd.read_csv(data_path,
                         header = None,
                         delim_whitespace = True,
                         names = headers)
    
    print("MPG Types Before:\n", df_mpg.dtypes, "\n")
    
    df_mpg = df_mpg.replace("?", np.NaN)
    
    print("Dim Before:\n", df_mpg.shape, "\n")
    
    df_mpg = df_mpg.dropna()
    
    print("Dim After:\n", df_mpg.shape, "\n")
    
    df_mpg_np = df_mpg.values
    
    mpg_data = df_mpg_np[:, 0:-1]
    mpg_targets = df_mpg_np[:, -1]
    
    mpg_data[:, 3] = label_encoder.fit_transform(mpg_data[:, 3])
    
    print("MPG After:\n", df_mpg_np, "\n")
    
    return mpg_data, mpg_targets
    
    
def create_training_set(data, targets):
    x_train, x_test, y_train, y_test = train_test_split(data, targets,
                                                        test_size = .3, random_state = 15)
    
    print("Data training:\n", x_train, "\n")
    print("Data targets:\n", y_train, "\n")
    
    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled


def run_knn(k, X, y):
    # Returns a list
    k_fold = KFold(len(y), n_folds = 10, random_state = 15, shuffle = True)
    
    classifier = KNeighborsClassifier(n_neighbors=k)
    
    predictions = cross_val_predict(classifier, X, y, cv = k_fold, n_jobs = 1)
    
    accuracy = cross_val_score(classifier, X, y, cv = k_fold, n_jobs = 1).mean()
    
    return predictions, accuracy
    
 
def main():
    
    # Get car data
    car_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    car_data, car_targets = load_car_data(car_path)
    
    car_predict, car_accuracy = run_knn(3, car_data, car_targets)
    
    print("Car Accuracy: {0:.0f}%\n".format(car_accuracy * 100))
    
    autism_path = "Autism-Adult-Data.csv"
    autism_data, autism_targets = load_autism_data(autism_path)
    
    autism_predict, autism_accuracy = run_knn(3, autism_data, autism_targets)
    
    print("Autism Accuracy: {0:.0f}%\n".format(autism_accuracy * 100))
    
    mpg_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    mpg_data, mpg_targets = load_mpg_data(mpg_path)
    
    mpg_predict, mpg_accuracy = run_knn(3, mpg_data, mpg_targets)
    
    print("MPG Accuracy: {0:.0f}%\n".format(mpg_accuracy * 100))
 
 
if __name__ == "__main__":
    main()
    
