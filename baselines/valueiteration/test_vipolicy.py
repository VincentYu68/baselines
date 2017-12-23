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

def main():
    path = 'data/value_iter_swingup_discrete'

    env = gym.make('DartCartPole-v1')
    env.env.disableViewer=False
    obs_disc = bin_disc([[50, 1, -1], [50, 0.5, -0.5], [50, 10, -10], [50, 20, -20]])
    act_disc = bin_disc([[50, -1, 1]])
    obs_disc_dim = 50 * 50 * 50 * 50
    act_disc_dim = 10

    dyn_model = joblib.load(path+'/dyn_model.pkl')
    policy = joblib.load(path+'/policy.pkl')


    for traj in range(10):
        env.reset()
        cur_state = obs_disc(env.env.state_vector())

        for step in range(500):
            cur_state, rew = advance_dyn_model(dyn_model, cur_state, policy[int(cur_state)])

            map_state = obs_disc.get_midstate(cur_state)
            env.env.set_state_vector(map_state)
            env.render()
            time.sleep(0.01)



if __name__ == '__main__':
    main()