import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = input_height-self.kernel_size+1
        output_width = input_width-self.kernel_size+1

        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        #Width dimensions. outChannels, inChannels, kernelSize, kernelSize

        for i in range(output_height):
            for j in range(output_width):
                # batchSize, inChannels, kernel_size, kernel_size
                patch = A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                Z[:, :, i, j] = np.tensordot(patch, self.W, axes=([1, 2, 3] ,[1, 2, 3]))

        Z += self.b.reshape(1, -1, 1, 1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Save original dimensions before padding
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        input_height = output_height + self.kernel_size - 1
        input_width = output_width + self.kernel_size - 1

        # 1. Calculate dLdb
        # Sum over batch (axis 0), height (axis 2), and width (axis 3)
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # 2. Pad dLdZ into a NEW variable for dLdA
        pad_size = self.kernel_size - 1
        padded_dLdZ = np.pad(
            dLdZ, 
            pad_width=((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
            mode='constant', 
            constant_values=0
        )
        
        flipped_w = np.flip(self.W, axis=(2, 3))

        # Initialize output tensors
        dLdA = np.zeros((batch_size, self.in_channels, input_height, input_width))
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        # ---------------------------------------------------------
        # 3. Calculate dLdA
        # ---------------------------------------------------------
        for i in range(input_height):
            for j in range(input_width):
                # Slice the PADDED gradient. 
                # Shape: (batch_size, out_channels, kernel_size, kernel_size)
                patch = padded_dLdZ[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                
                # patch: sum over out_channels (axis 1) and spatial (axes 2, 3)
                # flipped_w: sum over out_channels (axis 0) and spatial (axes 2, 3)
                # Leaves behind: batch_size and in_channels
                dLdA[:, :, i, j] = np.tensordot(patch, flipped_w, axes=([1, 2, 3], [0, 2, 3]))

        # ---------------------------------------------------------
        # 4. Calculate dLdW
        # ---------------------------------------------------------
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                # Slice the original input A. 
                # Shape: (batch_size, in_channels, output_height, output_width)
                patch = self.A[:, :, i : i + output_height, j : j + output_width]
                
                # dLdZ: sum over batch (0), height (2), width (3)
                # patch: sum over batch (0), height (2), width (3)
                # Leaves behind: out_channels from dLdZ, in_channels from patch
                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, patch, axes=([0, 2, 3], [0, 2, 3]))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d_stride1 and Downsample2d instances
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # 1. Pad the input appropriately using np.pad() function
        # We pad ONLY the height (axis 2) and width (axis 3)
        if self.pad > 0:
            A = np.pad(
                A, 
                pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 
                mode='constant', 
                constant_values=0
            )

        # 2. Call Conv2d_stride1
        Z_without_resampling = self.conv2d_stride1.forward(A)

        # 3. downsample
        Z = self.downsample2d.forward(Z_without_resampling)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # 1. Call downsample2d backward
        Z_without_resampling = self.downsample2d.backward(dLdZ)

        # 2. Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(Z_without_resampling)

        # 3. Unpad the gradient
        # We slice off the padding from the top/bottom (axis 2) and left/right (axis 3)
        # Note: We must check if pad > 0, because slicing with :-0 returns an empty array!
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad : -self.pad, self.pad : -self.pad]

        return dLdA
