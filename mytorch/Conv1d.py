# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        batch_size, in_channels, input_size = A.shape
        # print("A.shape", A.shape, "batch_size, in_channels, input_size")
        output_size = input_size - self.kernel_size + 1
        # print("output_size", output_size)
        Z = np.zeros((batch_size, self.out_channels, output_size))
        for b in range(batch_size):
            for filter in range(self.out_channels):
                    for i in range(output_size): # index at current filter
                        w = self.W[filter]
                        extracted_A = A[b][:, i:i+self.kernel_size]
                        # print("w.shape, extracted_A.shape", w.shape, extracted_A.shape, self.W.shape, self.A.shape, self.kernel_size)
                        Z[b][filter][i] = np.vdot(w, extracted_A) + self.b[filter]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # print("dLdZ_orig.shape", dLdZ.shape)
        batch_size, in_channels, input_size = self.A.shape
        batch_size, out_channels, output_size = dLdZ.shape

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        # broadcast dLdZ across the output_size dim and duplicate it in_channel # of times to match dLdW
        dLdZ = np.repeat(dLdZ, self.in_channels, axis=1).reshape((batch_size, out_channels, in_channels, output_size))


        self.dLdW = np.zeros(self.W.shape) # (out_channel, in_channel, kernel_size)
        for b in range(batch_size):
            for c in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(self.kernel_size):
                        extracted_dLdZ = dLdZ[b][c][ic]
                        extracted_A = self.A[b][ic][i:i+output_size]
                        
                        self.dLdW[c][ic][i] += np.vdot(extracted_dLdZ, extracted_A)

        # print("dLdZ.shape", dLdZ.shape)

        padded_dLdZ = np.zeros((batch_size, out_channels, in_channels, 2*(self.kernel_size-1) + output_size))
        for b in range(batch_size):
            for c in range(self.out_channels):
                for ic in range(self.in_channels):
                    padded_dLdZ[b][c] = np.pad(dLdZ[b][c],((0, 0),(self.kernel_size-1, self.kernel_size-1)),'constant')

        flipped_W = np.flip(self.W, axis=2) # flip along kernel dimension

        dLdA = np.zeros(self.A.shape)
        # print("A.shape", self.A.shape)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(input_size): # convolved across each input channel separately
                        # print("self.kernel_size", self.kernel_size, "output size", output_size, "input_size", input_size)
                        # print("first", padded_dLdZ[b][oc][ic][i:i+self.kernel_size].shape, "second", flipped_W[oc][ic].shape, "padded_dLdZ", padded_dLdZ.shape, i, i+self.kernel_size)
                        dLdA[b][ic][i] += np.vdot(padded_dLdZ[b][oc][ic][i:i+self.kernel_size], flipped_W[oc][ic])
                        
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        Z = self.conv1d_stride1.forward(A)
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        out = self.downsample1d.backward(dLdZ)
        # print(out.shape, "before stride back")
        dLdA = self.conv1d_stride1.backward(out)

        # Call Conv1d_stride1 backward

        return dLdA
