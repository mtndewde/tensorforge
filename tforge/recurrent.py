from .unit import *
from .dense import *
import tensorflow as tf


class BasicRNNCell(Unit, tf.contrib.rnn.RNNCell):

    def __init__(self, recurrent_layer):
        self._recurrent_layer = recurrent_layer
        self._state_size = self._recurrent_layer.output_dim

    @property
    def recurrent_layer(self):
        return self._recurrent_layer

    @classmethod
    def from_description(cls, n_in, n_hid, act, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            recurrent_layer = DenseLayer.from_description(n_in+n_hid, n_hid, act, scope)
        return cls(recurrent_layer)

    @property
    def variables(self):
        return self.recurrent_layer.variables

    @property
    def state_size(self):
        return self.recurrent_layer.output_dim

    @property
    def output_size(self):
        return self.recurrent_layer.output_dim

    def process(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn_output") as scope:
            outputs = self.recurrent_layer.process(tf.concat([inputs, state[0]], 1), scope)
        return outputs, (outputs,)

    def to_dictionary(self, session):
        return {"recurrent_layer": self.recurrent_layer.to_dictionary(session)}

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            recurrent_layer = DenseLayer.from_dictionary(data_dict["recurrent_layer"], scope)
        return cls(recurrent_layer)

pass


class BasicRNN(StatefulUnit):

    def __init__(self, cell, hid_state):
        super().__init__(cell, (hid_state,))

    @classmethod
    def from_cell(cls, cell, batch_size, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            hid_state = tf.get_variable(
                "hidden_state",
                shape=[batch_size, cell.state_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )
        return cls(cell, hid_state)

    @property
    def hidden_state(self):
        return self.state[0]

    def to_dictionary(self, session):
        return {"cell": self.cell.to_dictionary(session), "hiddenstate": session.run(self.hidden_state)}

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            cell = BasicRNNCell(BasicRNNCell.from_dictionary(data_dict["cell"]))
            hid_state = tf.get_variable(
                "hidden_state",
                shape=data_dict["hiddenstate"],
                dtype=tf.float32,
                initializer=tf.constant_initializer(data_dict["hiddenstate"]),
                trainable=False
            )
        return cls(cell, hid_state)


pass


class LSTMCell(Unit, tf.contrib.rnn.RNNCell):

    def __init__(self, forget_gate, input_gate, input_extractor, output_gate, output_extractor):
        self._forget_gate = forget_gate
        self._input_gate = input_gate
        self._input_extractor = input_extractor
        self._output_gate = output_gate
        self._output_extractor = output_extractor
        self._state_size = output_extractor.output_dim, input_extractor.output_dim

    @classmethod
    def from_description(cls, n_in, n_hid, n_cell, scope=None):
        with tf.variable_scope(scope, "lstm_cell"):
            forget_gate = DenseLayer.from_description(n_in + n_hid, n_cell, None, "forget_gate")
            input_gate = DenseLayer.from_description(n_in + n_hid, n_cell, tf.sigmoid, "input_gate")
            input_extractor = DenseLayer.from_description(n_in + n_hid, n_cell, tf.tanh, "input_extractor")
            output_gate = DenseLayer.from_description(n_in + n_hid, n_hid, tf.sigmoid, "output_gate")
            output_extractor = DenseLayer.from_description(n_cell, n_hid, tf.tanh, "output_extractor")
        return cls(forget_gate, input_gate, input_extractor, output_gate, output_extractor)

    @property
    def variables(self):
        return self.forget_gate.variables + self.input_gate.variables + self.input_extractor.variables + \
               self.output_gate.variables + self.output_extractor.variables

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self.output_extractor.output_dim

    @property
    def forget_gate(self):
        return self._forget_gate

    @property
    def input_gate(self):
        return self._input_gate

    @property
    def input_extractor(self):
        return self._input_extractor

    @property
    def output_gate(self):
        return self._output_gate

    @property
    def output_extractor(self):
        return self._output_extractor

    def process(self, inputs, state, scope=None):
        with tf.variable_scope(scope, "lstm_cell_output"):
            hidden_state, cell_state = state
            hidden_inputs = tf.concat([inputs, hidden_state], axis=1)
            cell_state = cell_state * tf.sigmoid((self.forget_gate.process(hidden_inputs) + 1))
            cell_state += self.input_gate.process(hidden_inputs) * self.input_extractor.process(hidden_inputs)
            hidden_state = self.output_gate.process(hidden_inputs) * self.output_extractor.process(cell_state)
        return hidden_state, (hidden_state, cell_state)

    def to_dictionary(self, session):
        return {
            "forget_gate": self.forget_gate.to_dictionary(session),
            "input_gate": self.input_gate.to_dictionary(session),
            "input_extractor": self.input_extractor.to_dictionary(session),
            "output_gate": self.output_gate.to_dictionary(session),
            "output_extractor": self.output_extractor.to_dictionary(session)
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, "lstm_cell"):
            forget_gate = DenseLayer.from_dictionary(data_dict["forget_gate"], "forget_gate")
            input_gate = DenseLayer.from_dictionary(data_dict["input_gate"], "input_gate")
            input_extractor = DenseLayer.from_dictionary(data_dict["input_extractor"], "input_extractor")
            output_gate = DenseLayer.from_dictionary(data_dict["output_gate"], "output_gate")
            output_extractor = DenseLayer.from_dictionary(data_dict["output_extractor"], "output_extractor")
        return cls(forget_gate, input_gate, input_extractor, output_gate, output_extractor)


pass


class LSTM(StatefulUnit):

    def __init__(self, cell, hid_state, cell_state):
        super().__init__(cell, (hid_state, cell_state))

    @classmethod
    def from_cell(cls, cell, batch_size, scope=None):
        with tf.variable_scope(scope, default_name="lstm"):
            n_hid, n_cell = cell.state_size
            init = tf.zeros_initializer()
            hid_state = tf.get_variable("hidden_state", [batch_size, n_hid], initializer=init, trainable=False)
            cell_state = tf.get_variable("cell_state", [batch_size, n_cell], initializer=init, trainable=False)
        return cls(cell, hid_state, cell_state)

    @property
    def hidden_state(self):
        return self.state[0]

    @property
    def cell_state(self):
        return self.state[1]

    def to_dictionary(self, session):
        return {
            "cell": self.cell.to_dictionary(session),
            "hiddenstate": session.run(self.hidden_state),
            "cellstate": session.run(self.cell_state)
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="lstm"):
            cell = LSTMCell.from_dictionary(data_dict["cell"], "lstm_cell")
            hid_state = tf.get_variable(
                "hidden_state",
                shape=data_dict["hiddenstate"].shape,
                initializer=tf.constant_initializer(data_dict["hiddenstate"]),
                trainable=False
            )
            cell_state = tf.get_variable(
                "cell_state",
                shape=data_dict["cellstate"].shape,
                initializer=tf.constant_initializer(data_dict["cellstate"]),
                trainable=False
            )
            lstm = cls(cell, hid_state, cell_state)
        return lstm


pass