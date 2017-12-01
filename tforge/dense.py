from .unit import *
import tensorflow as tf


class DenseLayer(Unit):

    def __init__(self, weights, bias, act):
        self._weights = weights
        self._bias = bias
        self._act = act
        self._variables = [self._weights, self._bias]
        self._input_dim, self._output_dim = weights.get_shape().as_list()

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def variables(self):
        return self._variables

    @property
    def activation(self):
        return self._act

    @property
    def input_dim(self):
        return  self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def to_dictionary(self, session):
        save_dict = session.run({"weights": self._weights, "bias": self._bias})
        save_dict["activation"] = self.activation.__name__ if self.activation is not None else "none"
        return save_dict

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        act = {"sigmoid": tf.sigmoid, "tanh": tf.tanh, "relu": tf.nn.relu, "none": None}[data_dict["activation"]]
        weights = data_dict["weights"]
        bias = data_dict["bias"]
        return cls.from_arrays(weights, bias, act, scope)

    @classmethod
    def from_description(cls, n_in, n_hid, act, scope=None):
        with tf.variable_scope(scope, default_name="dense_layer"):
            weights = tf.get_variable("weights", shape=[n_in, n_hid])
            bias = tf.get_variable("bias", shape=[n_hid])
        return cls(weights, bias, act)

    @classmethod
    def from_arrays(cls, weights_array, bias_array, act, scope=None):
        with tf.variable_scope(scope, default_name="dense_layer"):
            weights = tf.get_variable(
                "weights",
                shape = weights_array.shape,
                initializer = tf.constant_initializer(weights_array)
            )
            bias = tf.get_variable(
                "bias",
                shape = bias_array.shape,
                initializer = tf.constant_initializer(bias_array)
            )
        return cls(weights, bias, act)

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="dense_layer_output"):
            if self._bias is not None:
                output = tf.nn.xw_plus_b(inputs, self._weights, self._bias)
            else:
                output = tf.matmul(inputs, self._weights)
            output = self.activation(output) if self._act is not None else output
        return output

pass


class DenseStackedLayers(StackedUnits):

    @classmethod
    def from_description(cls, n_in, n_hids, acts, scope=None):
        dims = [n_in] + n_hids
        params = zip(dims[:-1], dims[1:], acts)
        with tf.variable_scope(scope, "dense_stacked_layers"):
            layers = [DenseLayer.from_description(n_in, n_hid, act, scope="dense_layer_%i" % (i,))
                      for i, (n_in, n_hid, act) in enumerate(params)]
        return cls(layers)

    def to_dictionary(self, session):
        return {"denselayer_%i" % (i + 1,): l.to_dictionary(session) for i, l in enumerate(self.units)}

    @classmethod
    def from_dictionary(cls, data_dict):
        ordered_keys = ["denselayer_%i" % (i + 1,) for i in range(len(data_dict.keys()))]
        return cls([DenseLayer.from_dictionary(data_dict[n]) for n in ordered_keys])

pass