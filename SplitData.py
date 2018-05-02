import pandas as pd
import pdb
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
        temp.append(0.0)
        for i in range(1, len(columns)):
            #if row[columns[i]] == 1:
            temp.append(row[i])
        temp.append(0.0)
        y.append(temp) #, [0.0]*(len(columns)-1) + [1.0]])

   
    #m = MultiLabelBinarizer()
    #Y = m.fit_transform(y)
    #pdb.set_trace()

    #Y = np.expand_dims(y,1)
    Y = y
    X = df["Post"]


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    Z_train = []
    W_train = []
    #pdb.set_trace()
    for line in Y_train:
            Z_train.append([[1.0] + [0.0]*len(columns), line])
            W_train.append([line, [0.0]*len(columns) + [1.0]])
    Y_train = np.array(Z_train)
    W_train = np.array(W_train)
    Z_test = []
    for line in Y_test:
            Z_test.append([line, [0.0]*len(columns) + [1.0]])
    Y_test = np.array(Z_test)
    #pdb.set_trace()
    return X_train, X_test, Y_train, Y_test, W_train, columns

