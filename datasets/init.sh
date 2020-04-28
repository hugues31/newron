#!/bin/bash
cd ./datasets

# Download Wine Quality (white) dataset
wine_url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv";
curl -X GET $wine_url > ./winequality-white.csv;

# Download MNIST Fashion dataset
training_images="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
training_labels="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
test_images="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
test_labels="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
mkdir fashion_mnist;

curl -X GET $training_images > ./fashion_mnist/train-images-idx3-ubyte.gz;
gzip -d fashion_mnist/train-images-idx3-ubyte.gz;

curl -X GET $training_labels > ./fashion_mnist/train-labels-idx1-ubyte.gz;
gzip -d fashion_mnist/train-labels-idx1-ubyte.gz;

curl -X GET $test_images > ./fashion_mnist/t10k-images-idx3-ubyte.gz;
gzip -d fashion_mnist/t10k-images-idx3-ubyte.gz;

curl -X GET $test_labels > ./fashion_mnist/t10k-labels-idx1-ubyte.gz;
gzip -d fashion_mnist/t10k-labels-idx1-ubyte.gz;
