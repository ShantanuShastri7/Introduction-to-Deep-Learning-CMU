import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A_shape = A.shape
        batch_size, in_channels, input_width, input_height = A.shape
        out_channels = in_channels
        self.output_width = input_width - self.kernel +1
        self.output_height = input_height - self.kernel +1
        Z = np.zeros((batch_size, out_channels, self.output_height, self.output_width))

        self.max_indices = np.zeros((batch_size, out_channels, self.output_height, self.output_width), dtype=int)

        for i in range(self.output_height):
            for j in range(self.output_width):
                # Extract the patch
                patch = A[:, :, i : i + self.kernel, j : j + self.kernel]
                
                # Find the maximum value to pass forward
                Z[:, :, i, j] = np.max(patch, axis=(2, 3))
                
                # To find the index, we must reshape the spatial dimensions to 1D
                patch_flat = patch.reshape(batch_size, in_channels, -1)
                
                # Store the flat index (0 through kernel*kernel - 1)
                self.max_indices[:, :, i, j] = np.argmax(patch_flat, axis=2)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros(self.A_shape)
        batch_size, in_channels, _, _ = self.A_shape
        
        # Create grid indices for the batch and channel dimensions.
        # This helps us route gradients to the correct batch/channel in a vectorized way.
        b_idx, c_idx = np.indices((batch_size, in_channels))

        for i in range(self.output_height):
            for j in range(self.output_width):
                # 1. Retrieve the flat indices for this specific spatial window
                flat_idx = self.max_indices[:, :, i, j]
                
                # 2. Use unravel_index to convert flat indices into 2D patch offsets
                # h_offset and w_offset will both be arrays of shape (batch_size, in_channels)
                h_offset, w_offset = np.unravel_index(flat_idx, (self.kernel, self.kernel))

                
                # 3. Add the gradient back to the original winning pixel
                # i + h_offset gets us the absolute row position in the original image
                # j + w_offset gets us the absolute col position in the original image
                #Using Advanced array indexing 
                dLdA[b_idx, c_idx, i + h_offset, j + w_offset] += dLdZ[b_idx, c_idx, i, j]

        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A_shape = A.shape
        batch_size, in_channels, input_height, input_width = A.shape
        out_channels = in_channels
        
        self.output_height = input_height - self.kernel + 1
        self.output_width = input_width - self.kernel + 1
        
        Z = np.zeros((batch_size, out_channels, self.output_height, self.output_width))

        for i in range(self.output_height):
            for j in range(self.output_width):
                # Extract the patch
                patch = A[:, :, i : i + self.kernel, j : j + self.kernel]
                
                # Compute the mean over the spatial dimensions (axes 2 and 3)
                Z[:, :, i, j] = np.mean(patch, axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros(self.A_shape)
        
        # Calculate the uniform gradient distribution factor: 1 / (K * K)
        grad_dist_factor = 1.0 / (self.kernel * self.kernel)

        for i in range(self.output_height):
            for j in range(self.output_width):
                # Grab the gradient for this specific window position
                # Shape: (batch_size, out_channels)
                grad = dLdZ[:, :, i, j]
                
                # Reshape it to (batch_size, out_channels, 1, 1) so it broadcasts across the patch
                grad_reshaped = grad[:, :, None, None]
                
                # Add the evenly scaled gradient back to every pixel in the patch
                dLdA[:, :, i : i + self.kernel, j : j + self.kernel] += grad_reshaped * grad_dist_factor

        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Initialize the stride1 and downsample instances
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # 1. Apply the pooling operation
        Z_stride1 = self.maxpool2d_stride1.forward(A)
        
        # 2. Downsample based on the stride
        Z = self.downsample2d.forward(Z_stride1)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # 1. Backpropagate through the downsampler (inflates the gradient)
        delta_out = self.downsample2d.backward(dLdZ)
        
        # 2. Backpropagate through the pooling math
        dLdA = self.maxpool2d_stride1.backward(delta_out)
        
        return dLdA


class MeanPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Initialize the stride1 and downsample instances
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # 1. Apply the pooling operation
        Z_stride1 = self.meanpool2d_stride1.forward(A)
        
        # 2. Downsample based on the stride
        Z = self.downsample2d.forward(Z_stride1)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # 1. Backpropagate through the downsampler (inflates the gradient)
        delta_out = self.downsample2d.backward(dLdZ)
        
        # 2. Backpropagate through the pooling math
        dLdA = self.meanpool2d_stride1.backward(delta_out)
        
        return dLdA