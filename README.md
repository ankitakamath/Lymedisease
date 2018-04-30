# Lymedisease
Keras model for Lymedisease analysis project

Requirements:
1. Install python3.6
2. Install packages of python like numpy, pandas,sklearn
3. Install keras package as well.

**********************************************************************************************************************
The repository consists of following python files:
1. SplitData.py : This file is used to split the dataset into training and testing dataframes.
2. seq2seq.py : This file is used to run the sequence to sequence LSTM model. This file consists of decode_sequence
                function which helps to predict and get the output in readable form.
3. encoder_decoder_model.py : This file contains the actual encoder-decoder model implementation.
4. evaluateModel.py : This file contains the precision, recall and accuracy calculation for each frames.
5. lyme_sequential : This file contains the implementation of lstm sequential model.


The lyme_sequential.py and seq2seq.py calls the evaluateModel.py for evaluation.
**********************************************************************************************************************
Steps to run the entire project:

1. Run ./lymedisease.sh - This will run the encoder-decoder model as well as sequential model.
2. If you get permission denied error while running previous command use : chmod u+x lymedisease.sh

**********************************************************************************************************************
Steps to run only Encoder- decoder model:

1. Run python3 SplitData.py
2. Run python3 seq2seq.py

**********************************************************************************************************************
Steps to run only sequential model:

1. Run python3 lyme_sequential.py

**********************************************************************************************************************

Note : The output may be different each time it is executed. This is because of the small size of dataset. Some of the classes
might not be trained. Since the dataset is unbalanced, hence the output may be biased towards certain classes.
