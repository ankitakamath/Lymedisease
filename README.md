# Lymedisease
Keras model for Lymedisease analysis project

Requirements:
1. Install python3.6
2. Install packages of python like numpy, pandas,sklearn
3. Install keras package as well.

**********************************************************************************************************************
The repository consists of 4 python files:
1. SplitData.py : This file is used to split the dataset into training and testing csv files.
2. seq2seq.py : This file is used to run the sequence to sequence LSTM model. This file consists of decode_sequence
                function which helps to get the output in readable form.
3. encoder_decoder_model.py : This file contains the actual encoder-decoder model implementation.
4. evaluateModel.py : This file contains the precision, recall and accuracy calculation for each frames.

**********************************************************************************************************************
Steps to run the project:
1. Change the path for glove dataset.
2. Run python3 SplitData.py
3. Change the path in seq2seq.py for train_dataset.csv and test_dataset.csv
4. Run python3 seq2seq.py

