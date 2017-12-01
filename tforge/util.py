from .unit import *
import tensorflow as tf


class Reshaper(Unit):

    def __init__(self, target_shape, name=None):
        self._target_shape = target_shape
        self._output_dim = self._target_shape[-1]

    @property
    def output_dim(self):
        return self._output_dim

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="reshape"):
            output = tf.reshape(inputs, shape=self._target_shape)
        return output


pass


class BatchFlattener(Reshaper):

    def __init__(self, batch_shape):
        super().__init__([-1] + batch_shape[-1:])
        self._batch_shape = batch_shape[:-1] + [-1]
        self._inverse = Reshaper(self._batch_shape)

    @classmethod
    def from_tensor(cls, batch):
        return cls(batch.get_shape().as_list())

    @property
    def inverse(self):
        return self._inverse


pass