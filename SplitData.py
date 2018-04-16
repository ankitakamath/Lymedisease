import pandas as pd
from sklearn.model_selection import train_test_split
import re

def removeStopWords(df):
    df = re.sub(r'^https?\/\/.*[\r\n]*', '', df, flags=re.MULTILINE)
    df = re.sub(r'^http?\/\/.*[\r\n]*', '', df, flags=re.MULTILINE)
    stopwords = ["i", "know", "still", "have", "cannot", "my", "and", "as", "if", "i'm", "a", "the", "with", "of",
                 "all", "for", "some", "too", "doc", "did.", "i've", "years,", "p", "k", "ct,", "h", "j", "who",
                 "at", "was", "is", "other", "even", "they","but", "myself", "be", "am", "an", "do", "not", "take",
                 "there", "are", "that","seems","to","which","ok","on","did","has","me","or","nt","you","are","in","a",
                 "he","oh","c","lol","l",'f',"just","many","out","way","after","your","seem","how","had","so","it's",
                 "although","sure"]
    arr = df.split()
    st = ""
    for i in arr:
        if i not in stopwords:
            st += i + " "
    df = st
    return df

def cleanData():
    #
    columns = ["Post", "Seek", "medical_condition", "medical_test", "medication", "progress",
               "failure", "insurance",  "diet", "exercise", "ask_for_advice", "other"]
    characters_to_remove = ['.','(',')',',','•',':','[',']','\\/','!','?','\\"','-','~','|','0','1','2','3','4','5','6','7',
                            '8','9','&','*','+',';','>','`']

    df = pd.read_csv("Data/dataset.csv", delimiter="^")
    col = list(df.columns.values)
    frames = []
    for index,row in df.iterrows():
        z = []
        for i in range(len(columns)):
            if row[columns[i]] == 1:
                z.append(columns[i])
        frames.append(','.join(z))
    df["Frames"] = frames
    df = df[df["Frames"] != ""]
    df["Post"] = df["Post"].apply(lambda m :''.join([c for c in m if c not in characters_to_remove]))
    # df["Post"] = df["Post"].apply(removeStopWords)
    df = df[df["Post"] != ""]
    return df

def splitData():
    df = cleanData()
    df = df[df["Post"].map(len) < 1000]
    train_data,test_data = train_test_split(df,test_size = 0.2)

    train_data.to_csv("Data/train_dataset.csv",sep="^")
    test_data.to_csv("Data/test_dataset.csv",sep="^")

splitData()