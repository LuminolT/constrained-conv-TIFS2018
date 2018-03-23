# Source code for: "Constrained Convolutional Neural Networks: A New approach Towards General Purpose Image Manipulation Detection"
by Belhassen Bayar and Matthew C. Stamm

Deparment of Electrical and Computer Engineering

Drexel University - Philadelphia, PA, USA

## About

This repository contains pycaffe scripts for general purpose image manipulation detection using constrained convolutional neural network (CNN), i.e., MISLnet architecture. The functions within this repository perform the following tasks:

- **training_mislnet.py:** Train a CNN architecture associated with a constrained convolutional layer. This scripts uses prototxt files, namely `solver_mislnet.prototxt` (training hyper-parameters) and `train_val_mislnet.prototxt` (CNN layers).
- **testing_mislnet.py**: Test a trained CNN using the caffe MISLnet model `mislnet_six_classes.caffemodel` that we provide along with `deploy_mislnet.prototxt` file. The MISLnet caffe model has been trained with 1.2M image patches of size 256x256 pixels created by the five different types of image processing defined in the paper. Class 0 corresponds to unaltered patches and the remaining classes are labeled with respect to the order we fllowed in Table III of the paper.
- **create_forgery_lmdb.py:** Create lmdb testing data using the list of 334 images `imglst_test.dmp` from Dresden Image Database that we selected to create 50K testing image patches as decribed in the paper. This code creates test_lmdb data under the root folder `caffe_scripts`. Also, this scripts returns the confusion matrix of the CNN.
- **deep_features_ert.py:** Extract training and testing deep features (second fully-connected layer after activation) from MISLnet caffe model `mislnet_six_classes.caffemodel` using the `deploy_mislnet.prototxt` file. Next, train and test an extremely randomized trees (ERT) classifier using the extracted deep features. This script returns also the testing accuracy along with the confusion matrix for both SoftMax-based and ERT-based CNN.