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

    def _init(self, input, output, hid_size, num_hid_layers, dropout = 0.1, input_placeholder=None):
        sequence_length = None

        self.params = []

        if input_placeholder is None:
            ob = U.get_placeholder(name="input", dtype=tf.float32, shape=[sequence_length] + [input,])
        else:
            ob = input_placeholder
        #training_flag = U.get_placeholder(name="training_flag", dtype=tf.bool, shape=[sequence_length]+[1,])

        last_out = ob
        for i in range(num_hid_layers):
            last_out, w, b = U.dense_wparams(last_out, hid_size, "fc%i"%(i+1), weight_init=U.normc_initializer(1.0))
            last_out = tf.tanh(last_out)
            self.params.append([w,b])
        mean, w, b = U.dense_wparams(last_out, output, "final", U.normc_initializer(1.0))
        self.params.append([w, b])
        self._pred = U.function([ob], [mean])
        self.mean = mean

    def get_symbolic_output(self, input):
        last_out = input
        for i in range(len(self.params)-1):
            pm = self.params[i]
            last_out = tf.matmul(last_out, pm[0]) + pm[1]
            last_out = tf.tanh(last_out)
        ret = tf.matmul(last_out, self.params[-1][0]) + self.params[-1][1]
        return ret

    def pred(self, ob):
        out =  self._pred(ob[None])
        return out[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

