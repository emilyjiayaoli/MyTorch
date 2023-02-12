import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:

    def forward(self, Z):
        self.A = 1/(1+np.exp(-Z)) 
        return self.A

    def backward(self):
        dAdZ = self.A - (self.A*self.A)
        return dAdZ

class Tanh:

    def forward(self, Z):
        self.A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        return self.A

    def backward(self):
        # we don't have Z, the orig input, but we know A = tanh(Z) so we just replace that tanh^2 w A^2
        dAdZ = 1-(self.A**2) # tanh′(x) = 1 − tanh^2(x).
        return dAdZ

class ReLU:
    
    def forward(self, Z):
        self.A = np.zeros((Z.shape[0], Z.shape[1]))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i][j] > 0:
                    self.A[i][j] = Z[i][j]
        return self.A

    def backward(self):
        dAdZ = np.zeros((self.A.shape[0], self.A.shape[1]))
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if self.A[i][j] > 0:
                    dAdZ[i][j] = 1
        return dAdZ
