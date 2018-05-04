#!/bin/sh

echo "Spliting the data"
python3 SplitData.py

echo "Running Encoder decoder model"
python3 seq2seq.py

echo "Running sequential model"
python3 lyme_sequential.py
