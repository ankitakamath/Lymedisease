import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

columns = ["Post", "Seek", "medical_condition", "medical_test", "medication", "insurance",
           "diet", "exercise", "ask_for_advice", "other"]
df = pd.read_csv("Data/dataset.csv", delimiter="^")

df[["Seek","medical_condition", "medical_test", "medication", "insurance",
                "diet","exercise", "ask_for_advice", "other"]] = df[["Seek","medical_condition", "medical_test", "medication", "insurance",
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

X = df["Post"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)


def LSTMmodel():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256)(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(9)(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


model = LSTMmodel()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr = 0.001), metrics=['accuracy'])

history = model.fit(sequences_matrix, Y_train, batch_size=3, epochs=10,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
#pyplot.show()
pyplot.savefig("validation_loss_seq2seq.pdf")
pyplot.clf()


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)


pred =  model.predict(test_sequences_matrix)
output = []
groundtruth = []

for i in range(len(Y_test)):
    print(pred[i])

accr = model.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))