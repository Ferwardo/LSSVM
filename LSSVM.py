import tensorflow as tf


class LSSVM:
    def __init__(self, Beta=None, Omega=None, P_inv=None, X_pv=None, Y_pv=None, zeta=tf.Variable([]), config=None):
        # Initialise parameters of the model
        self.Beta = Beta
        self.Omega = Omega
        if self.Omega is not None:
            self.Omega_inv = tf.linalg.inv(Omega)
        self.P_inv = P_inv
        self.P_inv_prev = P_inv
        self.X_pv = X_pv
        self.Y_pv = Y_pv
        self.zeta = zeta

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
        self.Omega_inv = tf.linalg.inv(self.Omega)

        # Compute Omega_tm for the rest of the calculations
        Omega_tm = self.__gen_kernel_matrix(X_init, X_pv, self.config["sigma"])

        # Compute P_inv with ((Omega_tm'*Omega_tm)+C*Omega_mm)^-1
        if C is None:
            C = self.config["C"]

        self.P_inv = tf.linalg.inv(
            tf.matmul(Omega_tm, Omega_tm, transpose_a=True) + tf.scalar_mul(C, self.Omega))

        # Compute Beta with P_inv . Omega_tm' . Y_init
        self.Beta = tf.linalg.matvec(tf.matmul(self.P_inv, Omega_tm, transpose_b=True), Y_init)

        # Compute the cost zeta
        tf.norm(Y_init - tf.linalg.matvec(Omega_tm, self.Beta) ** 2 + tf.scalar_mul(C, tf.tensordot(
            tf.linalg.matvec(self.Omega, self.Beta), self.Beta, 1)))

    def normal(self, X, Y):
        # Some setup for later
        # cost = []
        # cost[self.config["Ninit"]] = self.zeta
        self.P_inv_prev = self.P_inv

        for n in range(self.config["Ninit"] + 1, X.shape[0] + 1):
            sigma = self.__gen_kernel_matrix(X[n, :], self.X_pv, sigma=self.config["sigma"])
            epsilon = Y[n] - tf.matmul(sigma, self.Beta, transpose_a=True)  # y_t-sigma^T(x_t).Beta
            Delta = 1 + tf.matmul(tf.matmul(sigma, self.P_inv, transpose_a=True),
                                  sigma)  # 1+sigma^T(x_t).P_inv.sigma(x_t)

            # Compute P_inv with P_inv - (P_inv.(sigma(x_t).sigma(x_t)').P_inv)/Delta
            sigmaProduct = tf.matmul(sigma, sigma, transpose_b=True)
            self.P_inv -= tf.matmul(self.P_inv, tf.matmul(sigmaProduct, self.P_inv)) / Delta

            # Compute Beta with Beta - (epsilon/Delta).P_inv.sigma(x_t)
            self.Beta -= tf.matmul((epsilon / Delta), tf.matmul(self.P_inv, sigma))

            # Compute the cost.
            self.zeta += tf.matmul(epsilon, epsilon, transpose_b=True) / Delta

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
                # Compute the kernel value with the following formula: kerval=||X(n,:)-X_t(nt,:)||²
                kerval = tf.norm(X[n, :] - X_t[nt, :]) ** 2

                # Compute the current row for Omega
                Omega[n, nt].assign(tf.exp(-kerval / sigma))

        return Omega
