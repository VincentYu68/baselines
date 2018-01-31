__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json

from utils import *

def main():
    path = 'data/value_iter_hopper_discrete'

    env = gym.make('DartCartPoleSwingUp-v1')
    env.env.disableViewer = False

    dyn_model = joblib.load(path + '/dyn_model.pkl')
    policy = joblib.load(path + '/policy.pkl')
    [Vfunc, obs_disc, act_disc] = joblib.load(path + '/ref_policy_funcs.pkl')

    reward_mat = np.zeros((obs_disc.bin_num, act_disc.bin_num))
    policy_mat = np.zeros((obs_disc.bin_num, act_disc.bin_num))
    vf_val = np.zeros(100)
    test_state = [0, 0, 0, 0]
    for i in range(100):
        test_state[1] += i * 6.0/100
        if obs_disc(test_state) in Vfunc:
            vf_val[i] = Vfunc[obs_disc(test_state)]

    for s in dyn_model:
        for a in dyn_model[s]:
            avg_r = 0.0
            for sn in dyn_model[s][a]:
                avg_r += dyn_model[s][a][sn][0] * dyn_model[s][a][sn][1]
            reward_mat[int(s)][int(a)] = avg_r
        policy_mat[int(s)][int(policy[s])] = 1.0

    plt.matshow(reward_mat.T)
    plt.colorbar()
    plt.matshow(policy_mat.T)
    plt.colorbar()
    plt.figure()
    plt.plot(vf_val)
    plt.show()

if __name__ == '__main__':
    main()