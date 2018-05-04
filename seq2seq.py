from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from encoder_decoder_model import  encoder_decoder_model
import numpy as np
from evaluateModel import evaluate
import pandas as pd

columns = ["Post", "Seek", "medical_condition", "medical_test", "medication", "insurance",
           "diet", "exercise", "ask_for_advice", "other"]

# df_train = pd.read_csv("Data/train.csv", delimiter="^")
# df_test = pd.read_csv("Data/test.csv",delimiter ="^")

df_train = pd.read_pickle("Data/train.pkl")
df_test = pd.read_pickle("Data/test.pkl")
Y_train = df_train["Frames_seq2seq"]
Y_test = df_test["Frames_seq2seq"]
X_train  = df_train["Post"]
X_test  = df_test["Post"]

Z_train = []
W_train = []
# pdb.set_trace()
for line in Y_train:
    print(line)
    Z_train.append([[1.0] + [0.0] * len(columns), line])
    W_train.append([line, [1.0] + [0.0] * len(columns)])
Y_train = np.array(Z_train)
decode_train = np.array(W_train)
Z_test = []
for line in Y_test:
    Z_test.append([line, [0.0] * len(columns) + [1.0]])
Y_test = np.array(Z_test)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

num_decoder_tokens = 11
encoder_model,decoder_model = encoder_decoder_model(sequences_matrix,Y_train, decode_train)

def decode_sequence(input_seq):
    # Encode the input as
    #  state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0,0,0] = 1.0
    output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
    print("Result")
    print(output_tokens[0][0])
    return output_tokens[0][0]



pred = []
actual = []

for seq_index in range(len(test_sequences_matrix)):
    # # Take one sequence (part of the training set)
    # # for trying out decoding.
    input_seq = test_sequences_matrix[seq_index: seq_index + 1]
    classes = decode_sequence(input_seq)
    pred.append(classes)
    actual.append(Y_test[seq_index][0])

evaluate(pred,actual,columns)

