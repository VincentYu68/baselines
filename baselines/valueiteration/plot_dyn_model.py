__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json

from utils import *

def main():
    path = 'data/value_iter_swingup_discrete'

    obs_disc = bin_disc([[10, 1, -1], [20, 6.0, -6.0], [5, 10, -10], [20, 20, -20]])
    act_disc = bin_disc([[10, 1.0, -1.0]])
    obs_disc_dim = 10 * 20 * 5 * 20
    act_disc_dim = 10

    dyn_model = joblib.load(path + '/dyn_model.pkl')
    policy = joblib.load(path + '/policy.pkl')

    reward_mat = np.zeros((obs_disc_dim, act_disc_dim))
    policy_mat = np.zeros((obs_disc_dim, act_disc_dim))

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
    plt.show()

if __name__ == '__main__':
    main()