import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        batch_size, in_channels, in_width = A.shape
        self.A_shape = (batch_size, in_channels, in_width)
        Z = np.reshape(A, (batch_size, -1))  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, self.A_shape)  # TODO

        return dLdA
