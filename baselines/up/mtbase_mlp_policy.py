from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

def mtbdense(x, mp, size, name, weight_init=None, bias=True):
    snetnum = mp.shape[1] + 1
    indiv_mps = tf.split(mp, np.int32([1]*(snetnum-1)), 1)
    w_base = tf.get_variable(name + "/w_base", [x.get_shape()[1], size], initializer=weight_init)

    w_nom = tf.get_variable(name + "/w_s0", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w_nom + w_base)
    for n in range(snetnum-1):
        w = tf.get_variable(name + "/w_s"+str(n+1), [x.get_shape()[1], size], initializer=weight_init)
        ret += indiv_mps[n] * tf.matmul(x, w - w_nom)

    if bias:
        b_base = tf.get_variable(name + "/b_base", [size], initializer=tf.zeros_initializer())
        b_nom = tf.get_variable(name + "/b_s0", [size], initializer=tf.zeros_initializer())
        ret += b_nom + b_base
        for n in range(snetnum - 1):
            b = tf.get_variable(name + "/b_s"+str(n+1), [size], initializer=tf.zeros_initializer())
            ret += indiv_mps[n] * (b-b_nom)
        return ret
    else:
        return ret

class MultiTaskBaseMlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, gmm_comp=1, mp_dim = 0, obs_name='ob', final_std=0.01):
        assert isinstance(ob_space, gym.spaces.Box)

        self.mp_dim = mp_dim

        self.pdtype = pdtype = make_pdtype(ac_space, gmm_comp)
        sequence_length = None
        print(self.pdtype)

        ob = U.get_placeholder(name=obs_name, dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        _, split_d = tf.split(ob, [ob_space.shape[0]-mp_dim, mp_dim], 1)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=(ob_space.shape[0],))

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        split_o, _ = tf.split(obz, [ob_space.shape[0] - mp_dim, mp_dim], 1)

        general_init_std = 1.0 / 2.0
        last_outvf = obz
        for i in range(num_hid_layers):
            last_outvf = tf.nn.tanh(U.dense(last_outvf, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(general_init_std)))
        self.vpred = U.dense(last_outvf, 1, "vffinal", weight_init=U.normc_initializer(general_init_std))[:,0]

        last_out = split_o
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(mtbdense(last_out, split_d, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(general_init_std)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            if gmm_comp == 1:
                mean = mtbdense(last_out, split_d, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(final_std/2.0))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer(0.0))
                pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)

        if gmm_comp == 1:
            self.pd = pdtype.pdfromflat(pdparam)
        else:
            self.pd = pdtype.pdfromflat([pdparam, gmm_comp])

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


