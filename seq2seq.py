from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from encoder_decoder_model import  encoder_decoder_model
from SplitData import SplitData
import numpy as np
from evaluateModel import evaluateModel

X_train, X_test, Y_train, Y_test,columns = SplitData()

num_decoder_tokens = 9
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

encoder_model,decoder_model = encoder_decoder_model(sequences_matrix,Y_train)

def decode_sequence(input_seq):
    # Encode the input as
    #  state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
    print("Result")
    print(output_tokens[0][0])



for seq_index in range(len(test_sequences_matrix)):
    # # Take one sequence (part of the training set)
    # # for trying out decoding.
    input_seq = test_sequences_matrix[seq_index: seq_index + 1]
    classes = decode_sequence(input_seq)


