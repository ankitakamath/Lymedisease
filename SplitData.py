import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

def SplitData():
    columns = ["Post", "Seek", "medical_condition", "medical_test", "medication", "insurance",
           "diet", "exercise", "ask_for_advice", "other"]
    df = pd.read_csv("Data/dataset.csv", delimiter="^")

    df[["Seek","medical_condition", "medical_test", "medication", "insurance",
                "diet","exercise", "ask_for_advice", "other"]] = df[["Seek","medical_condition",
                                                                     "medical_test", "medication", "insurance",
                "diet","exercise", "ask_for_advice", "other"]].astype(float).fillna(0.0)
    y_seq2seq = []
    y_sequential = []
    for index, row in df.iterrows():
        temp1 =[]
        temp = []
        temp.append(0.0)
        for i in range(1, len(columns)):
            #if row[columns[i]] == 1:
            temp.append(row[i])
            if row[columns[i]] == 1:
                temp1.append(columns[i])
        temp.append(0.0)
        y_seq2seq.append(temp) #, [0.0]*(len(columns)-1) + [1.0]])
        y_sequential.append(temp1)

   
    #m = MultiLabelBinarizer()
    #Y = m.fit_transform(y)
    #pdb.set_trace()

    #Y = np.expand_dims(y,1)
    Y = y_seq2seq
    df["Frames_seq2seq"] = Y
    df["Frames_sequential"] = y_sequential
    train,test = train_test_split(df, test_size=0.15)
    train.to_pickle("Data/train.pkl")
    test.to_pickle("Data/test.pkl")


SplitData()
