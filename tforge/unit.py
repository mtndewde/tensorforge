import tensorflow as tf
import h5py


class Unit(object):

    def process(self, *args, **kwargs):
        raise NotImplementedError("process is abstract")

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @property
    def variables(self):
        return []

    @property
    def initializers(self):
        return [v.initializer for v in self.variables]

    def to_dictionary(self, session):
        raise NotImplementedError("to_dictionary is abstract")

    @classmethod
    def from_dictionary(cls, data_dict):
        raise NotImplementedError("from_dictionary is abstract")

    @staticmethod
    def _write_dict_to_hdf_group(data_dict, group):
        for name, data in data_dict.items():
            if isinstance(data, dict):
                Unit._write_dict_to_hdf_group(data, group.create_group(name))
            else:
                group.create_dataset(name, data=data)

    @staticmethod
    def _load_hdf_group_to_dict(group):
        data_dict = {}
        for name, subgroup in group.items():
            if isinstance(subgroup, h5py.Dataset):
                data_dict[name] = subgroup.value
            else:
                data_dict[name] = Unit._load_hdf_group_to_dict(subgroup)
        return data_dict

    def save(self, session, file):
        data_dict = self.to_dictionary(session)
        with h5py.File(file, "w") as f:
            Unit._write_dict_to_hdf_group(data_dict, f)

    @classmethod
    def from_file(cls, file):
        with h5py.File(file, "r") as f:
            unit = cls.from_dictionary(Unit._load_hdf_group_to_dict(f))
        return unit

pass


class StackedUnits(Unit):

    def __init__(self, units):
        self._stacked_units = units

    @property
    def variables(self):
        return [v for u in self._stacked_units for v in u.variables]

    @property
    def units(self):
        return self._stacked_units

    def process(self, inputs):
        outputs = inputs
        for unit in self._stacked_units:
            outputs = unit.process(outputs)
        return outputs

pass


class StatefulUnit(Unit):

    def __init__(self, cell, state_tuple):
        self._cell = cell
        self._state = state_tuple
        self._reset_state = [tf.assign(s, tf.zeros(s.get_shape())) for s in self.state]

    @property
    def state(self):
        return self._state

    @property
    def cell(self):
        return self._cell

    @property
    def variables(self):
        return self.cell.variables + list(self.state)

    @property
    def reset_state(self):
        return self._reset_state

    def process(self, inputs, stateful=True, scope=None):
        with tf.variable_scope(scope, default_name="statefulunit_output"):
            outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.state, scope=scope)
            if stateful:
                with tf.control_dependencies([tf.assign(os, ns) for os, ns in zip(self.state, state)]):
                    outputs = tf.identity(outputs)
        return outputs

pass

