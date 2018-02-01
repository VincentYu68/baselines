__author__ = 'yuwenhao'

import gym
import sys, os, time
from baselines.common import set_global_seeds, tf_util as U
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from baselines.ppo1 import mlp_policy
from utils import *

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=3, gmm_comp=1)

def main():
    sess = tf.InteractiveSession()
    path = 'data/value_iter_pendulum_discrete'

    env = gym.make('DartPendulum-v1')
    env.env.disableViewer = False

    dyn_model = joblib.load(path + '/dyn_model.pkl')
    policy = joblib.load(path + '/policy.pkl')
    [Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn] = joblib.load(path + '/ref_policy_funcs.pkl')

    policy_params = joblib.load('data/ppo_DartPendulum-v12_vf_vanilla_2k/policy_params.pkl')
    ob_space = env.observation_space
    ac_space = env.action_space
    ct_policy = policy_fn("pi", ob_space, ac_space)

    U.initialize()

    cur_scope = ct_policy.get_variables()[0].name[0:ct_policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]

    for i in range(len(ct_policy.get_variables())):
        assign_op = ct_policy.get_variables()[i].assign(
            policy_params[ct_policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)


    s_list = list(Vfunc.keys())
    vf_disc = []
    vf_cont = []
    o = env.reset()
    d = False
    while not d:
        if obs_disc(state_filter_fn(env.env.state_vector())) not in Vfunc:
            vf_disc.append(-1)
        else:
            vf_disc.append(Vfunc[obs_disc(state_filter_fn(env.env.state_vector()))]*0.5)
        ac, vf = ct_policy.act(True, o)
        if np.random.random() < 0.1:
            ac = env.action_space.sample()
        vf_cont.append(vf)
        o, r, d, _ = env.step(ac)
    '''for samp in range(100):
        samp_id = np.random.randint(len(s_list))
        sampled_s = s_list[samp_id]
        vf_disc.append(Vfunc[sampled_s])

        ct_vfs = []
        for i in range(100):
            obs = state_filter_fn(obs_disc.samp_state(sampled_s))
            ct_vfs.append(ct_policy.act(False, obs)[1])
        vf_cont.append(np.mean(ct_vfs))'''

    plt.plot(vf_disc, label="disc vf")
    plt.plot(vf_cont, label="cont vf")
    plt.legend()

    # plot 2d vf if possible
    if len(env.env.state_vector()) == 2:
        disc_size = 100
        vf_2d_disc = np.zeros((disc_size, disc_size))
        vf_2d_cont = np.zeros((disc_size, disc_size))
        for r in range(disc_size):
            for c in range(disc_size):
                state = np.array([r*1.0/disc_size * np.pi, c*1.0/disc_size * 30-15])

                if obs_disc(state_filter_fn(state)) in Vfunc:
                    vf_2d_disc[r][c] = Vfunc[obs_disc(state_filter_fn(state))]

                env.env.set_state_vector(state)
                vf_2d_cont[r][c] = ct_policy.act(False, env.env._get_obs())[1]
        plt.figure()
        plt.matshow(vf_2d_disc)
        plt.colorbar()
        plt.title('disc vf')
        plt.matshow(vf_2d_cont)
        plt.colorbar()
        plt.title('cont vf')


    plt.show()

if __name__ == '__main__':
    main()