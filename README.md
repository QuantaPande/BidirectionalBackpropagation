# BidirectionalBackpropagation
An implementation of the bidirectional backpropagation algorithm outlined in the paper linked

Bidirectional backpropagation is a new way of training neural networks, which utilises the reverse network for increasing prediction strength with negligible increase in time complexity of training. 

Neural networks can be though of as set-level maps between the input space and the output space. We know that for a set level map, the inverse map always exists and maps from a set in the output space to a set in the input space. The paper ( http://sipi.usc.edu/~kosko/B-BP-SMC-Revised-13January2018.pdf ) outlines a way in which this inverse map can be used to increase the strength of the approximate map between the input and the output space.

The repository contains an implementation of bidirectional backpropagation using Tensorflow to simple MLP based architectures. It also contains an example code, with results of the inverse map for the MNIST Dataset.

## Demo images of the results of the reverse map on MNIST images:
