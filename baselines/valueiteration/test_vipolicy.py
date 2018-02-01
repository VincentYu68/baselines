#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import joblib
import tensorflow as tf
import numpy as np
from mpi4py import MPI
from utils import *
from baselines.ppo1 import mlp_policy
import tensorflow as tf
from baselines.common import set_global_seeds, tf_util as U
from gym import wrappers

np.random.seed(0)

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3, gmm_comp=1)

def main():
    path = 'data/value_iter_cartpole_discrete_v4'

    env = gym.make('DartCartPoleSwingUp-v1')
    env.seed(0)
    env.env.disableViewer = False

    record = False
    if record:
        env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True)
    else:
        env_wrapper = env

    dyn_model = joblib.load(path+'/dyn_model.pkl')
    policy = joblib.load(path+'/policy.pkl')
    [Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn] = joblib.load(path + '/ref_policy_funcs.pkl')

    print('model loading done')
    print(obs_disc.disc_scheme)


    for traj in range(1):
        env_wrapper.reset()
        s = state_filter_fn(env.env.state_vector())
        cur_state = obs_disc(s)
        total_rew = 0
        for step in range(500):
            #print(cur_state, policy[int(cur_state)])
            #print(dyn_model[cur_state][policy[int(cur_state)]])
            #if policy[int(cur_state)] > 5:
            #    print(dyn_model[cur_state])

            if np.random.random() < 0.999 and int(cur_state) in policy and policy[int(cur_state)] is not None:
                act = policy[int(cur_state)]
            else:
                act = np.random.randint(act_disc.bin_num)

            ob, rew, d, _ = env_wrapper.step(act_disc.samp_state(act))
            cur_state = obs_disc(state_filter_fn(env.env.state_vector()))
            #cur_state, rew = advance_dyn_model(dyn_model, cur_state, int(act))
            #map_state = obs_disc.get_midstate(cur_state)
            #env.env.set_state_vector(state_unfilter_fn(map_state))

            env_wrapper.render()
            #time.sleep(0.05)
            total_rew += rew
        print('Return: ', total_rew)




if __name__ == '__main__':
    main()