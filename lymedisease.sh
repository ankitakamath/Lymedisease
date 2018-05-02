#!/bin/sh
echo "Running Encoder decoder model"
python3 seq2seq.py

echo "Running sequential model"
python3 lyme_sequential.py
