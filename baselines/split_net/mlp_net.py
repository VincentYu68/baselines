from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

class MlpNet(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, input, output, hid_size, num_hid_layers):
        sequence_length = None

        ob = U.get_placeholder(name="input", dtype=tf.float32, shape=[sequence_length] + [input,])

        last_out = ob
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        mean = U.dense(last_out, output, "final", U.normc_initializer(0.01))
        self._pred = U.function([ob], [mean])
        self.mean = mean

    def pred(self, ob):
        out =  self._pred(ob[None])
        return out[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

