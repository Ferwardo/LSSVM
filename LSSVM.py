import tensorflow as tf


class LSSVM:
    def __init__(self, Beta=tf.Variable([]), Omega=tf.Variable([]), P_inv=tf.Variable([]), X_pv=tf.Variable([]),
                 zeta=tf.Variable([]), config=None):
        # Initialise parameters of the model
        self.Beta = Beta
        self.Omega = Omega
        self.Omega_inv = tf.inv(Omega)
        self.P_inv = P_inv
        self.X_pv = X_pv
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

    def compute(self, X_init, Y_init, X_pv=None, C=None):
        """
        Compute the model parameters from scratch
        :param X_init: The observations used to initialise the model
        :param X_pv: The prototype vectors used to initialise the model
        """

        # Compute Omega_mm and its inverse
        if X_pv is None:
            X_pv = self.X_pv

        self.Omega = self.__gen_kernel_matrix(X_pv, X_pv, self.config["sigma"])
        self.Omega_inv = tf.inv(self.Omega)

        # Compute Omega_tm for the rest of the calculations
        Omega_tm = self.__gen_kernel_matrix(X_init, X_pv, self.config["sigma"])

        # Compute P_inv with ((Omega_tm'*Omega_tm)+C*Omega_mm)^-1
        if C is None:
            C = self.config["C"]

        self.P_inv = tf.inv(
            tf.matmul(Omega_tm, Omega_tm, transpose_a=True) + tf.scalar_mul(C, self.Omega))

        # Compute Beta with P_inv . Omega_tm' . Y_init
        self.Beta = tf.matmul(tf.matmul(self.P_inv, Omega_tm, transpose_b=True), Y_init)

        # Compute the cost zeta
        tf.norm(Y_init - tf.matmul(Omega_tm, self.Beta) ** 2 + tf.scalar_mul(C, tf.matmul(
            tf.matmul(self.Beta, self.Omega, transpose_a=True), self.Beta)))

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
        Omega = tf.Variable(tf.zeros([size_x[0], size_x_t[0]]))

        assert size_x[1] != size_x_t[1], "The matrices do not have the right size."

        for n in range(0, size_x[0]):
            for nt in range(0, size_x_t[0]):
                # Compute the kernel value with the following formula: kerval=||X(n,:)-X_t(nt,:)||²
                kerval = tf.norm(tf.subtract(X[n, :], X_t[nt, :])) ** 2

                # Compute the current row for Omega
                Omega[n, nt] = tf.exp(-kerval / sigma)

        return Omega
