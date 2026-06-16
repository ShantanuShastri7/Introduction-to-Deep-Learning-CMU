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
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        
        self.layers = [
            self.conv1, 
            ReLU(), 
            self.conv2, 
            ReLU(), 
            self.conv3, 
            Flatten()
        ]

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

        # ----------------------------------------------------
        # Layer 1 Weights: (192, 8) -> (8, 24, 8)
        # ----------------------------------------------------
        # 1 : Transpose the layer's weight matrix
        w1_t = w1.T
        # 2 : Reshape into (out_channels, kernel_size, in_channels) -> (8, 8, 24)
        w1_r = w1_t.reshape(8, 8, 24)
        # 3 : Transpose back into (out_channels, in_channels, kernel_size)
        self.conv1.conv1d_stride1.W = w1_r.transpose(0, 2, 1)

        # ----------------------------------------------------
        # Layer 2 Weights: (8, 16) -> (16, 8, 1)
        # ----------------------------------------------------
        w2_t = w2.T
        w2_r = w2_t.reshape(16, 1, 8)
        self.conv2.conv1d_stride1.W = w2_r.transpose(0, 2, 1)

        # ----------------------------------------------------
        # Layer 3 Weights: (16, 4) -> (4, 16, 1)
        # ----------------------------------------------------
        w3_t = w3.T
        w3_r = w3_t.reshape(4, 1, 16)
        self.conv3.conv1d_stride1.W = w3_r.transpose(0, 2, 1)

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
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        
        self.layers = [
            self.conv1, 
            ReLU(), 
            self.conv2, 
            ReLU(), 
            self.conv3, 
            Flatten()
        ]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            weights (list): Contains w1, w2, w3
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]

        # ----------------------------------------------------
        # Layer 1 Weights
        # ----------------------------------------------------
        w1_t = w1.T                                    # (8, 192)
        w1_r = w1_t.reshape(8, 8, 24)                  # (out_neurons, spatial, in_channels)
        w1_trans = w1_r.transpose(0, 2, 1)             # (out_neurons, in_channels, spatial) -> (8, 24, 8)
        self.conv1.conv1d_stride1.W = w1_trans[:2, :, :2] # Slice: (2, 24, 2)

        # ----------------------------------------------------
        # Layer 2 Weights
        # ----------------------------------------------------
        w2_t = w2.T                                    # (16, 8)
        w2_r = w2_t.reshape(16, 4, 2)                  # (out_neurons, spatial, in_channels)
        w2_trans = w2_r.transpose(0, 2, 1)             # (out_neurons, in_channels, spatial) -> (16, 2, 4)
        self.conv2.conv1d_stride1.W = w2_trans[:8, :, :2] # Slice: (8, 2, 2)

        # ----------------------------------------------------
        # Layer 3 Weights
        # ----------------------------------------------------
        w3_t = w3.T                                    # (4, 16)
        w3_r = w3_t.reshape(4, 2, 8)                   # (out_neurons, spatial, in_channels)
        w3_trans = w3_r.transpose(0, 2, 1)             # (out_neurons, in_channels, spatial) -> (4, 8, 2)
        self.conv3.conv1d_stride1.W = w3_trans[:4, :, :2] # Slice: (4, 8, 2)

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
            print(f"Before forward shape: {Z.shape}")
            Z = layer.forward(Z)
            print(f"After forward shape: {Z.shape}")
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
