import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

def SplitData():
    columns = ["Post", "Seek", "medical_condition", "medical_test", "medication", "insurance",
           "diet", "exercise", "ask_for_advice", "other"]
    df = pd.read_csv("Data/dataset.csv", delimiter="^")

    df[["Seek","medical_condition", "medical_test", "medication", "insurance",
                "diet","exercise", "ask_for_advice", "other"]] = df[["Seek","medical_condition",
                                                                     "medical_test", "medication", "insurance",
                "diet","exercise", "ask_for_advice", "other"]].astype(float).fillna(0.0)
    y = []
    for index, row in df.iterrows():
        temp = []
        for i in range(1, len(columns)):
            if row[columns[i]] == 1:
                temp.append(columns[i])
        y.append(temp)

    m = MultiLabelBinarizer()
    Y = m.fit_transform(y)
    Y = np.expand_dims(Y,1)
    X = df["Post"]


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
    return X_train, X_test, Y_train, Y_test,columns

