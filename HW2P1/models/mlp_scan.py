# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.layers = [] # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        
        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)

        self.conv1.conv1d_stride1.W = None
        self.conv2.conv1d_stride1.W = None
        self.conv3.conv1d_stride1.W = None

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method
        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.layers = [] # TODO: Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)
        #   4 : Slice the weight matrix and reduce to only the shared weights
        #   (hint: be careful, steps 1-3 are similar, but not exactly like in the simple scanning MLP)
        
        self.conv1.conv1d_stride1.W = None
        self.conv2.conv1d_stride1.W = None
        self.conv3.conv1d_stride1.W = None

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
