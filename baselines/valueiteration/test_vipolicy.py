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

np.random.seed(0)

def main():
    path = 'data/value_iter_truehopper_discrete'

    env = gym.make('DartHopper-v1')
    env.seed(0)
    env.env.disableViewer = False

    dyn_model = joblib.load(path+'/dyn_model.pkl')
    policy = joblib.load(path+'/policy.pkl')
    [Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn] = joblib.load(path + '/ref_policy_funcs.pkl')

    print('model loading done')
    print(obs_disc.disc_scheme)

    for traj in range(10):
        env.reset()
        s = state_filter_fn(env.env.state_vector())
        cur_state = obs_disc(s)

        for step in range(500):
            #print(cur_state, policy[int(cur_state)])
            #print(dyn_model[cur_state][policy[int(cur_state)]])
            #if policy[int(cur_state)] > 5:
            #    print(dyn_model[cur_state])

            if np.random.random() < 0.999 and int(cur_state) in policy and policy[int(cur_state)] is not None:
                act = policy[int(cur_state)]
            else:
                acts = list(dyn_model[cur_state].keys())
                if len(acts) == 0:
                    break
                else:
                    act = acts[np.random.randint(len(acts))]
            cur_state, rew = advance_dyn_model(dyn_model, cur_state, int(act))

            map_state = obs_disc.get_midstate(cur_state)
            env.env.set_state_vector(state_unfilter_fn(map_state))
            env.render()
            time.sleep(0.25)




if __name__ == '__main__':
    main()