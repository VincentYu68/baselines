from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, gmm_comp=1):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space, gmm_comp)
        sequence_length = None
        print(self.pdtype)

        ob = U.get_placeholder(name="ob", dtype=tf.float64, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        #obz = ob
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            if gmm_comp == 1:
                mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(1.0))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer(), dtype=tf.float64)
                pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
            else:
                means = U.dense(last_out, (pdtype.param_shape()[0] - gmm_comp)//2, "polfinal", U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd",
                                         initializer=tf.constant(np.ones((1, (pdtype.param_shape()[0] - gmm_comp)//2), dtype=np.float64)*(-1.0)))
                weights = tf.nn.softmax(U.dense(last_out, gmm_comp, "gmmweights", U.normc_initializer(0.01)))
                pdparam = U.concatenate([means, means*0.0+logstd, weights], axis=1)
        elif gmm_comp == 1:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(1.0))
        else:
            meanstd = U.dense(last_out, pdtype.param_shape()[0]-gmm_comp, "polfinal", U.normc_initializer(0.01))
            weights = tf.nn.softmax(U.dense(last_out, gmm_comp, "gmmweights", U.normc_initializer(0.01)))
            pdparam = U.concatenate([meanstd, weights], axis=1)

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

