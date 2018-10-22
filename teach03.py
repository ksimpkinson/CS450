import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def main():
    headers = ["age", "workclass", "fnlwgt", "education", 
    "education-num", "marital-status", "occupation", "relationship", 
    "race", "sex", "capital-gain", "capital-loss", "hours-per-week", 
    "native-country", "target"]

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None, names=headers)

    print(df.dtypes)

    df = df.values

    label_encoder = LabelEncoder()
    df[:, 1] = label_encoder.fit_transform(df[:, 1])

    # df[:, 3] = label_encoder.fit_transform(df[:, 3])

    print(df[:30][:])

    return


if __name__ == '__main__':
    main()