import tensorflow as tf


class Kroma:
    def __init__(self, config):
        self._weights_indices = tf.placeholder(tf.int64)
        self._weights_values = tf.placeholder(config.data_type)
        self._weights_shape = tf.placeholder(tf.int64)
        self._b = tf.placeholder(config.data_type)
        self._initial_x = tf.placeholder(config.data_type)

        weights = tf.SparseTensor(self.weights_indices, self.weights_values, self.weights_shape)
        x = self.initial_x
        for i in range(config.num_iterations):
            # Jacobi iteration
            x = self.b - tf.sparse_tensor_dense_matmul(weights, x)
        self._final_x = x

    @property
    def weights_indices(self):
        return self._weights_indices

    @property
    def weights_values(self):
        return self._weights_values

    @property
    def weights_shape(self):
        return self._weights_shape

    @property
    def b(self):
        return self._b

    @property
    def initial_x(self):
        return self._initial_x

    @property
    def final_x(self):
        return self._final_x
