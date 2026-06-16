import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Wout = self.upsampling_factor*(A.shape[2]-1)+1
        Z = np.zeros((A.shape[0], A.shape[1], Wout))
        Z[:, :, ::self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = dLdZ[:, :, ::self.upsampling_factor]  # TODO
        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.Win = A.shape[2]
        if self.downsampling_factor>0:
            Z = A[:, :, ::self.downsampling_factor] 
            return Z
        
        return A

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.Win))  # TODO
        dLdA[:, :, ::self.downsampling_factor] = dLdZ
        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        Wout = self.upsampling_factor*(A.shape[3]-1)+1
        Hout = self.upsampling_factor*(A.shape[2]-1)+1
        Z = np.zeros((A.shape[0], A.shape[1], Hout, Wout))
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]  

        return dLdA


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.Hin = A.shape[2]
        self.Win = A.shape[3]
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.Hin, self.Win))
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ
        
        return dLdA
