#!/bin/bash
cd ./datasets

# Download Wine Quality (white) dataset
wine_url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv";
curl -X GET $wine_url > ./winequality-white.csv;
