import pandas as pd
import numpy as np

from encoder_decoder_model import seq2seqmodel
from evaluateModel import  evaluateModel

# read the train and validate csv files
train_data = pd.read_csv("Data/train_dataset.csv",delimiter="^")
test_data = pd.read_csv("Data/test_dataset.csv",sep="^")

# dictionaries for word to int and int to word
wordToInt = dict()
intToWord = dict()


# encoded target characters for learning
target_characters = dict()
target_characters["\t"] = 0
target_characters["\n"] = 1
target_characters['Seek'] = 2
target_characters['medical_condition']=  3
target_characters['medical_test'] = 4
target_characters['medication'] = 5
target_characters['progress'] = 6
target_characters['failure'] = 7
target_characters['insurance'] = 8
target_characters['diet'] = 9
target_characters['exercise'] = 10
target_characters['ask_for_advice'] = 11
target_characters['other'] = 12

# reverse characters for output
reverse_target_characters = dict()
reverse_target_characters[0] = '\t'
reverse_target_characters[1] = '\n'
reverse_target_characters[2] = 'Seek'
reverse_target_characters[3] =  'medical_condition'
reverse_target_characters[4] =  'medical_test'
reverse_target_characters[5] = 'medication'
reverse_target_characters[6] = 'progress'
reverse_target_characters[7] = 'failure'
reverse_target_characters[8] = 'insurance'
reverse_target_characters[9] = 'diet'
reverse_target_characters[10] = 'exercise'
reverse_target_characters[11] = 'ask_for_advice'
reverse_target_characters[12] = 'other'

def encodeLabelData(df):
    df = convertLabelData(df)
    m = []
    m.append(0)
    m.extend(df)
    m.append(1)
    return m

def convertLabelData(df):
    arr = df.split(",")
    m = []
    for y in range(len(arr)):
        m.append(target_characters[arr[y]])
    df = m
    return df

def isNotEmpty(s):
    try:
        return bool(s["Post"] and s["Post"].strip())
    except:
        False

train_data["Post_empty"] = train_data.apply(isNotEmpty, axis=1)
train_data = train_data[train_data["Post_empty"] == True]

test_data["Post_empty"] = test_data.apply(isNotEmpty, axis=1)
test_data = test_data[test_data["Post_empty"] == True]
train_post = train_data["Post"].tolist()

train_data["Frames"] = train_data["Frames"].apply(encodeLabelData)
frames = train_data["Frames"].tolist()

test_post_list = test_data["Post"].tolist()
groundtruth = test_data["Frames"].tolist()

count = 0

for post in train_post:
    post = post.split()
    for x in post:
        if x not in wordToInt.keys():
            wordToInt[x] = count
            intToWord[count] = x
            count = count + 1

num_encoder_tokens = count + 1
max_encoder_seq_length = max([len(txt.split()) for txt in train_post])
num_decoder_tokens = 14
max_decoder_seq_length = max([len(x) for x in frames])


encoder_input_data = np.zeros((len(train_data), max_encoder_seq_length), dtype='float32')
decoder_target_data = np.zeros((len(train_data), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(train_data), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (inp, out) in enumerate(zip(train_post, frames)):
    inp = inp.split()
    for t, word in enumerate(inp):
        encoder_input_data[i, t] = wordToInt[word]
    for t, char in enumerate(out):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, char] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, char] = 1.

model, encoder_model, decoder_model = seq2seqmodel(num_encoder_tokens, num_decoder_tokens,wordToInt,encoder_input_data,decoder_input_data,
                                                   decoder_target_data)


def decode_sequence(input_seq):
    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_characters['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_characters[sampled_token_index]
        decoded_sentence += sampled_char + ","
        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == '\n':
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

test_post = np.zeros((len(test_post_list),max_encoder_seq_length), dtype='float32')

for i,inp in enumerate(test_post_list):
    inp = inp.split()
    for t,word in enumerate(inp):
        if word in wordToInt.keys() and t < max_encoder_seq_length:
            test_post[i,t] = wordToInt[word]



output = []
for seq_index in range(len(test_post_list)):
    input_seq = test_post[seq_index :seq_index +1]
    print(test_post_list[seq_index])
    truth = groundtruth[seq_index]
    print("Ground truth ",truth)
    decoded_sentence = decode_sequence(input_seq)
    d = list(decoded_sentence.split(","))
    decoded_sentence = ",".join(list(set(d)))
    decoded_sentence = "".join(decoded_sentence.split("\t"))
    output.append(decoded_sentence)
    print("decoded sentence ",decoded_sentence)


print("results")

#print(classification_report(output,groundtruth))

evaluateModel(output,groundtruth)
