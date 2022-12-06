import tensorflow as tf


class LSSVM:
    def __init__(self):
        self.Beta = tf.Variable([])
        self.Omega = tf.Variable([])
        self.P = tf.Variable([])
