import numpy as np
from resampling import *
from Conv1d import *
from Conv2d import *


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d instance
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # 1. Upsample the input (inserts zeros between elements)
        A_upsampled = self.upsample1d.forward(A)

        # 2. Call Conv1d_stride1 on the upsampled data
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in the reverse order of the forward pass
        
        # 1. Propagate gradient through the convolution
        delta_out = self.conv1d_stride1.backward(dLdZ)

        # 2. Propagate gradient through the upsampler
        dLdA = self.upsample1d.backward(delta_out)

        return dLdA


class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d_stride1 and upsample2d instance
        self.upsample2d = Upsample2d(upsampling_factor)
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # 1. Upsample the spatial dimensions
        A_upsampled = self.upsample2d.forward(A)

        # 2. Apply the convolution
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call backward in the reverse order of the forward pass
        
        # 1. Propagate gradient through the convolution
        delta_out = self.conv2d_stride1.backward(dLdZ)

        # 2. Propagate gradient through the upsampler
        dLdA = self.upsample2d.backward(delta_out)

        return dLdA