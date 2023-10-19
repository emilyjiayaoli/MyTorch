import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = np.ones((self.A.shape[0], 1))  # TODO
        self.C = np.ones((self.A.shape[1], 1))  # TODO
        self.n = self.A.shape[0]
        self.c = self.A.shape[1]

        se = (A-Y)*(A-Y)  # TODO
        sse = self.N.T @ se @ self.C  # TODO
        mse = sse[0][0]/(2 * self.n * self.c)  # TODO

        return mse

    def backward(self):

        dLdA = (self.A-self.Y)/self.n * self.c * 0.25

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = self.A.shape[0]  # TODO
        C = self.A.shape[1]  # TODO

        Ones_C = np.ones((C,1))  # TODO
        Ones_N = np.ones((N,1))  # TODO

        self.softmax = np.exp(A)/np.sum(np.exp(A), axis=1, keepdims=True) # (N, C) # TODO
        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C # N, 1 # TODO
        sum_crossentropy = Ones_N.T @ crossentropy  # TODO
        L = sum_crossentropy / N

        return L[0][0]

    def backward(self):

        dLdA = self.softmax - self.Y  # Do not need to backprop through - log, just use logits # TODO

        return dLdA
