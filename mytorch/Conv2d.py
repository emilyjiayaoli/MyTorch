import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

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
        # print("A.shape",A.shape, "self.kernel_size", self.kernel_size)
        # print("W.shape", self.W.shape) # out_channels, in_channels, kernel_size, kernel_size

        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))
        for b in range(batch_size):
            for filter in range(self.out_channels):
                W = self.W[filter]
                for h in range(output_height):
                    for w in range(output_width): # index at current filter
                        extracted_A = A[b][:, h:h+self.kernel_size, w:w+self.kernel_size]
                        # print("w.shape, extracted_A.shape", W.shape, extracted_A.shape, h+self.kernel_size)
                        Z[b][filter][h][w] = np.vdot(W, extracted_A) + self.b[filter]
        return Z
    

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # self.dLdW = None  # TODO
        # self.dLdb = None  # TODO
        # dLdA = None  # TODO

        batch_size, in_channels, input_height, input_width = self.A.shape
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        # print(self.b.shape, dLdZ.shape)

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        # print(self.b.shape, self.dLdb.shape)

        # broadcast dLdZ across the output_size dim and duplicate it in_channel # of times to match dLdW
        # dLdZ = np.repeat(dLdZ, self.in_channels, axis=1).reshape((batch_size, out_channels, in_channels, output_size))


        self.dLdW = np.zeros(self.W.shape) # (out_channel, in_channel, kernel_size, kernel_size)
        for b in range(batch_size):
            for c in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            extracted_dLdZ = dLdZ[b][c]
                            extracted_A = self.A[b][ic][i:i+output_height, j:j+output_width]
                            # print("extracted_dLdZ, extracted_A", extracted_dLdZ.shape, extracted_A.shape)
                            
                            self.dLdW[c][ic][i][j] += np.vdot(extracted_dLdZ, extracted_A)

        # print("dLdZ.shape", dLdZ.shape)

        padded_dLdZ = np.zeros((batch_size, out_channels, in_channels, 2*(self.kernel_size-1)+output_height, 2*(self.kernel_size-1)+output_width))
        for b in range(batch_size):
            for c in range(self.out_channels):
                for ic in range(self.in_channels):
                    padded_dLdZ[b][c] = np.pad(dLdZ[b][c],((self.kernel_size-1, self.kernel_size-1),(self.kernel_size-1, self.kernel_size-1)),'constant')

        flipped_W = np.flip(self.W, axis=2) # flip along height kernel dimension
        flipped_W = np.flip(flipped_W, axis=3) # flip along width kernel dimension

        dLdA = np.zeros(self.A.shape)
        # print("A.shape", self.A.shape)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(input_height): # convolved across each input channel separately
                        for j in range(input_width): # convolved across each input channel separately
                            # print("self.kernel_size", self.kernel_size, "output size", output_size, "input_size", input_size)
                            # print("first", padded_dLdZ[b][oc][ic][i:i+self.kernel_size].shape, "second", flipped_W[oc][ic].shape, "padded_dLdZ", padded_dLdZ.shape, i, i+self.kernel_size)
                            dLdA[b][ic][i][j] += np.vdot(padded_dLdZ[b][oc][ic][i:i+self.kernel_size, j:j+self.kernel_size], flipped_W[oc][ic])
                        
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,
                 kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        out = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(out)

        return dLdA
