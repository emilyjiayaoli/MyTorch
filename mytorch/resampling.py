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
        batch_size, in_channels, input_width = A.shape

        Z = np.zeros((batch_size, in_channels, self.upsampling_factor*(input_width-1)+1))
        for b in range(batch_size):
            for c in range(in_channels):
                counter = 0
                for w in range(0, Z.shape[2], self.upsampling_factor):
                    Z[b][c][w] = A[b][c][counter]
                    counter+=1
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, input_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, ((input_width-1)//self.upsampling_factor)+1))

        for b in range(batch_size):
            for c in range(in_channels):
                counter = 0
                for w in range(0, dLdZ.shape[2], self.upsampling_factor):
                    dLdA[b][c][counter] = dLdZ[b][c][w]
                    counter+=1
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
        batch_size, in_channels, input_width = A.shape

        # determining output width after downsampling based on divisibility
        if input_width % self.downsampling_factor == 0:
            w_out = int(input_width/self.downsampling_factor) 
        else:
            w_out = int(input_width//self.downsampling_factor) + 1

        Z = np.zeros((batch_size, in_channels, w_out))
        # print("Z.shape, A.shape",Z.shape, A.shape)

        # populate Z
        for b in range(batch_size):
            for c in range(in_channels):
                counter = 0
                for w in range(0, input_width, self.downsampling_factor):
                    # print(counter, w, input_width,  self.downsampling_factor, w_out)
                    if counter < w_out:
                        Z[b][c][counter] = A[b][c][w]
                        counter+=1
        self.width_in = input_width
        # print(Z.shape, "Z.shape at the end of forward downsampling")
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, input_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.width_in))
        for b in range(batch_size):
            for c in range(in_channels):
                counter = 0
                for w in range(0, self.width_in, self.downsampling_factor):
                    # print(dLdZ[b][c][counter])
                    dLdA[b][c][w] = dLdZ[b][c][counter]
                    counter+=1
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

        batch_size, in_channels, input_height, input_width = A.shape

        Z = np.zeros((batch_size, in_channels, self.upsampling_factor*(input_height-1)+1, self.upsampling_factor*(input_width-1)+1))
        for b in range(batch_size):
            for c in range(in_channels):
                counter_h = 0
                for h in range(0, Z.shape[2], self.upsampling_factor):
                    counter_w = 0
                    for w in range(0, Z.shape[3], self.upsampling_factor):
                        Z[b][c][h][w] = A[b][c][counter_h][counter_w]
                        counter_w += 1
                    counter_h += 1
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, input_height, input_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, ((input_height-1)//self.upsampling_factor)+1, ((input_width-1)//self.upsampling_factor)+1))

        for b in range(batch_size):
            for c in range(in_channels):
                counter_h = 0
                for h in range(0, dLdZ.shape[2], self.upsampling_factor):
                    counter_w = 0
                    for w in range(0, dLdZ.shape[3], self.upsampling_factor):
                        dLdA[b][c][counter_h][counter_w] = dLdZ[b][c][h][w]
                        counter_w += 1
                    counter_h += 1
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
        batch_size, in_channels, input_height, input_width = A.shape

        if input_width % self.downsampling_factor == 0:
            w_out_width = int(input_width/self.downsampling_factor) 
        else:
            w_out_width = int(input_width//self.downsampling_factor) + 1

        if input_height % self.downsampling_factor == 0:
            w_out_height = int(input_height/self.downsampling_factor) 
        else:
            w_out_height = int(input_height//self.downsampling_factor) + 1


        # if (input_height) % 2 == 0:
        #     w_out_height = (input_height//self.downsampling_factor)
        # else:
        #     w_out_height = (input_height//self.downsampling_factor)+1
        # if (input_width) % 2 == 0:
        #     w_out_width = (input_width//self.downsampling_factor)
        # else:
        #     w_out_width = (input_width//self.downsampling_factor)+1

        Z = np.zeros((batch_size, in_channels, w_out_height, w_out_width))
        for b in range(batch_size):
            for c in range(in_channels):
                counter_h = 0
                for h in range(0, input_height, self.downsampling_factor):
                    counter_w = 0
                    for w in range(0, input_width, self.downsampling_factor):
                        Z[b][c][counter_h][counter_w] = A[b][c][h][w]
                        counter_w += 1
                    counter_h += 1

        self.height_in = input_height
        self.width_in = input_width
        
        return Z
    

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, input_height, input_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.height_in, self.width_in))
        for b in range(batch_size):
            for c in range(in_channels):
                counter_h = 0
                for h in range(0, self.height_in, self.downsampling_factor):
                    counter_w = 0
                    for w in range(0, self.width_in, self.downsampling_factor):
                        dLdA[b][c][h][w] = dLdZ[b][c][counter_h][counter_w]
                        counter_w += 1
                    counter_h += 1
        return dLdA
