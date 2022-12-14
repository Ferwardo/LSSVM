import json

import tensorflow as tf


class LSSVM:
    def __init__(self, Beta=None, Omega=None, P_inv=None, X_pv=None, Y_pv=None, zeta=None,
                 config=None):
        # Initialise parameters of the model
        self.Beta = Beta
        self.Omega = Omega
        if self.Omega is not None:
            self.Omega_inv = tf.linalg.pinv(Omega)
        self.P_inv = P_inv
        self.P_inv_prev = P_inv
        self.X_pv = X_pv
        self.Y_pv = Y_pv
        self.zeta = zeta
        self.costs = []

        if config is None:
            config = {
                "K": 2,
                "Ninit": 1,
                "PVinit": 1,
                "M": 1500,
                "C": 10,
                "sigma": 20,
                "threshold_type": 'fxd',
                "threshold_minimum": 0.05,
            }
        self.config = config

    # I/O functions for the federated learning
    def get_parameters(self, to_file=False):
        model = {
            "beta": {
                "shape": self.Beta.shape,
                "value": self.Beta.numpy().tolist()
            },
            "omega": {
                "shape": self.Omega.shape,
                "value": self.Omega.numpy().tolist()
            },
            "omega_inv": {
                "shape": self.Omega_inv.shape,
                "value": self.Omega_inv.numpy().tolist()
            },
            "p_inv": {
                "shape": self.P_inv.shape,
                "value": self.P_inv.numpy().tolist()
            },
            "x_pv": {
                "shape": self.X_pv.shape,
                "value": self.X_pv.numpy().tolist()
            },
            "y_pv": {
                "shape": self.Y_pv.shape,
                "value": self.Y_pv.numpy().tolist()
            },
            "zeta": {
                "shape": self.zeta.shape,
                "value": self.zeta.numpy().tolist()
            }
        }

        jsonString = json.dumps(model)
        if to_file:
            with open("fl_params.json", "w") as outfile:
                outfile.write(jsonString)

        return jsonString

    def get_federated_learning_params(self, as_json=False, to_file=False):
        if not as_json:
            return self.Beta

        jsonString = json.dumps({"beta": {
            "shape": self.Beta.numpy().shape,
            "value": self.Beta.numpy().tolist()
        }}, indent=2)

        if to_file:
            with open("fl_params.json", "w") as outfile:
                outfile.write(jsonString)

        return jsonString

    # Computation steps for the model itself
    def compute(self, X_init, Y_init, X_pv=None, Y_pv=None, C=None):
        """
        Compute the model parameters from scratch
        :param X_init: The observations used to initialise the model
        :param Y_init: The labels for the corresponding observations of X_init
        :param X_pv: The prototype vectors used to initialise the model, optional
        :param Y_init: The labels for the corresponding observations of X_pv, optional
        :param C: Regularisation parameters, optional
        """

        if self.X_pv is None:
            assert X_pv is not None, "Either pass the inital X_pv in the constructor or this function"
            self.X_pv = X_pv
        else:
            X_pv = self.X_pv

        if self.Y_pv is None:
            assert Y_pv is not None, "Either pass the inital Y_pv in the constructor or this function"
            self.Y_pv = Y_pv

        # Compute Omega_mm and its inverse
        self.Omega = self.__gen_kernel_matrix(X_pv, X_pv, self.config["sigma"])
        self.Omega_inv = tf.linalg.pinv(self.Omega)

        # Compute Omega_tm for the rest of the calculations
        Omega_tm = self.__gen_kernel_matrix(X_init, X_pv, self.config["sigma"])

        # Compute P_inv with ((Omega_tm'*Omega_tm)+C*Omega_mm)^-1
        if C is None:
            C = self.config["C"]

        self.P_inv = tf.linalg.pinv(
            tf.matmul(Omega_tm, Omega_tm, transpose_a=True) + tf.scalar_mul(C, self.Omega))

        # Compute Beta with P_inv . Omega_tm' . Y_init
        self.Beta = tf.linalg.matvec(tf.matmul(self.P_inv, Omega_tm, transpose_b=True), Y_init)
        self.Beta = tf.reshape(self.Beta, (self.Beta.shape[0], 1))

        # Compute the cost zeta
        self.zeta = (tf.norm(Y_init - tf.matmul(Omega_tm, self.Beta)) ** 2) + \
                    tf.scalar_mul(C, tf.matmul(self.Beta, tf.matmul(self.Omega, self.Beta), transpose_a=True))

    def normal(self, X, Y):
        """
        Does a normal training step
        :param X: The observations to train on
        :param Y: The corresponding class labels of X
        """

        for n in range(self.config["Ninit"], X.shape[0]):
            self.P_inv_prev = self.P_inv

            # Get the current row and make a matrix out of it so it can do the calculation of the kernel later
            # for all the prototype vectors
            x = tf.reshape(tf.tile(X[n, :], [self.X_pv.shape[0]]), (self.X_pv.shape[0], X.shape[1]))
            sigma = self.__gen_kernel_matrix(x, self.X_pv, sigma=self.config["sigma"])

            # The valus of sigma are repeated the same amount as there are prototype vector, take only the first one and
            # make a 1D Tensor (read vector) out of it.
            sigma = tf.reshape(tf.convert_to_tensor(sigma.numpy()[0]), (sigma.numpy()[0].shape[0], 1))

            epsilon = Y.numpy()[n] - tf.linalg.matmul(sigma, self.Beta, transpose_a=True)  # y_t-sigma^T(x_t).Beta
            Delta = 1 + tf.matmul(sigma, tf.matmul(self.P_inv,
                                                   sigma), transpose_a=True)  # 1+sigma^T(x_t).P_inv.sigma(x_t)

            # Compute P_inv with P_inv - (P_inv.(sigma(x_t).sigma(x_t)').P_inv)/Delta
            sigmaProduct = tf.matmul(sigma, sigma, transpose_b=True)
            self.P_inv -= tf.matmul(tf.matmul(self.P_inv, sigmaProduct), self.P_inv) / Delta

            # Compute Beta with Beta + (epsilon/Delta).P_inv.sigma(x_t)
            # The tf.reshape is so that the epsilon/Delta is a scalar instead of a 1D tensor
            self.Beta += tf.matmul(tf.math.scalar_mul(tf.reshape(epsilon / Delta, []), self.P_inv), sigma)

            # Compute the cost.
            self.zeta += tf.math.multiply(epsilon, epsilon) / Delta
            if n < len(self.costs):  # Add zeta to the model costs.
                self.costs[n] += self.zeta
            else:
                self.costs.append(self.zeta)

    def predict(self, x):
        """
        Predicts the label of the given observation
        :param x: A single vector with an observation to be classified.
        :return: The predicted class label
        """
        sigma = self.__gen_kernel_matrix(tf.reshape(tf.tile(x, [self.X_pv.shape[0]]), (self.X_pv.shape[0], x.shape[0])),
                                         self.X_pv, sigma=self.config["sigma"])
        sigma = tf.reshape(tf.convert_to_tensor(sigma.numpy()[0]), (sigma.numpy()[0].shape[0], 1))
        temp = tf.linalg.matmul(self.Beta, sigma, transpose_a=True)

        return tf.math.sign(tf.linalg.matmul(self.Beta, sigma, transpose_a=True)).numpy()[0][0]

    # Private helper functions
    def __gen_kernel_matrix(self, X, X_t, sigma):
        """
        Computes the RBF kernel matrix between X and Xt given kernel bandwidth sigma
        :param X: An N times D data matrix
        :param X_t: An N_t times D data matrix
        :param sigma: The kernel bandwidth
        :return: The N times N_t kernel matrix.
        """

        size_x = X.shape
        size_x_t = X_t.shape
        Omega = tf.Variable(tf.zeros((size_x[0], size_x_t[0]), dtype=tf.float64))

        assert size_x[1] == size_x_t[1], "The matrices do not have the right size."

        for n in range(0, size_x[0]):
            for nt in range(0, size_x_t[0]):
                # Compute the kernel value with the following formula: kerval=||X(n,:)-X_t(nt,:)||??
                kerval = tf.norm(X[n, :] - X_t[nt, :]) ** 2

                # Compute the current row for Omega
                Omega[n, nt].assign(tf.exp(-kerval / sigma))

        return Omega
