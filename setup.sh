#!/bin/bash 

#conda install tensorflow
#cd finetune_alexnet_with_tensorflow/
#git checkout Hashing_model
mkdir -p model_data/pretrained_alexnet
mkdir -p model_data/trained_weights
cd model_data/pretrained_alexnet/
wget www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
cd ../../
mkdir data
cd data/
wget https://github.com/thulab/DeepHash/releases/download/v0.1/cifar10.zip
unzip cifar10.zip
rm -rf cifar10.zip
#conda install opencv

