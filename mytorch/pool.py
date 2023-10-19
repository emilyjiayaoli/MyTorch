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

        self.A = A

        batch_size, in_channels, input_height, input_width = A.shape
        # print("A.shape",A.shape, "self.kernel_size", self.kernel_size)
        # print("W.shape", self.W.shape) # out_channels, in_channels, kernel_size, kernel_size

        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        self.maxIndexZ = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=np.int8)
        # print("self.kernel", self.kernel)
        for b in range(batch_size):
            for ic in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width): # index at current filter
                        # print(A.shape)
                        extracted_A = A[b][ic][h:h+self.kernel, w:w+self.kernel]

                        cur_max = np.nanmax(extracted_A)


                        id_max = np.array(np.where(extracted_A == cur_max))
                        # print(cur_max, id_max)
                        # print(extracted_A)
                        self.maxIndexZ[b][ic][h][w][0], self.maxIndexZ[b][ic][h][w][1] = int(h + id_max[0][0]), int(w + id_max[1][0])
                        # if (h + id_max[0][0])>output_height:
                            # print("checking",h, id_max[0][0], output_height)

                        # assert(extracted_A.shape == (self.kernel_size, self.kernel_size))
                        # raise ValueError
                        # print("w.shape, extracted_A.shape", W.shape, extracted_A.shape, h+self.kernel_size)
                        Z[b][ic][h][w] = cur_max
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # NOTE: didn't even use dLdZ to compute dLdA

        dLdA = np.zeros(self.A.shape)

        batch_size, in_channels, output_height, output_width, two = self.maxIndexZ.shape
        batch_size, in_channels, input_height, input_width = self.A.shape

        for b in range(batch_size):
            for ic in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        idx_x, idx_y = self.maxIndexZ[b][ic][h][w][0], self.maxIndexZ[b][ic][h][w][1]

                        # dLdA[b][ic][idx_x][idx_y] += self.A[b][ic][idx_x][idx_y] # wrong!
                        dLdA[b][ic][idx_x][idx_y] += dLdZ[b][ic][h][w] # right

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        batch_size, in_channels, input_height, input_width = A.shape

        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        for b in range(batch_size):
            for ic in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width): # index at current filter
                        # print(A.shape)
                        extracted_A = A[b][ic][h:h+self.kernel, w:w+self.kernel]

                        cur_mean = np.mean(extracted_A)
                        Z[b][ic][h][w] = cur_mean
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.A.shape)

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        batch_size, in_channels, input_height, input_width = self.A.shape

        for b in range(batch_size):
            for ic in range(in_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        for i in range(self.kernel):
                            for j in range(self.kernel):
                                dLdA[b][ic][h+i][w+j] += dLdZ[b][ic][h][w] / (self.kernel * self.kernel)# + 

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1

        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        out = self.downsample2d.backward(dLdZ)
        out = self.maxpool2d_stride1.backward(out)

        return out


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

        

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        out = self.downsample2d.backward(dLdZ)
        out = self.meanpool2d_stride1.backward(out)

        return out
