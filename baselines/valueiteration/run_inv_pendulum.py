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


def learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim):
    dyn_rwd_model = {}

    collected_data = []
    collected_steps = 0

    while collected_steps < 5000:
        d = False
        env.reset()
        while not d:
            bg_step = env.env.state_vector()
            act = env.action_space.sample()
            o,r,d,_ = env.step(act)
            ed_step = env.env.state_vector()
            collected_data.append([bg_step, act, ed_step, r])
            collected_steps += 1

    for trans in collected_data:
        if obs_disc(trans[0]) not in dyn_rwd_model:
            dyn_rwd_model[obs_disc(trans[0])] = {}
        if act_disc(trans[1]) not in dyn_rwd_model[obs_disc(trans[0])]:
            dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])] = {}
        if obs_disc(trans[2]) not in dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])]:
            dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])] = [0.0, 0.0]
        dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])][0] += 1
        dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])][1] += trans[3]

    for s in dyn_rwd_model.keys():
        for a in dyn_rwd_model[s].keys():
            occurence = 0
            for sn in dyn_rwd_model[s][a].keys():
                occurence += dyn_rwd_model[s][a][sn][0]
                dyn_rwd_model[s][a][sn][1] /= dyn_rwd_model[s][a][sn][0]
            for sn in dyn_rwd_model[s][a].keys():
                dyn_rwd_model[s][a][sn][0] /= occurence

    return dyn_rwd_model

def optimize_policy(dyn_rwd_model, gamma):
    Vfunc = {}
    for iter in range(3000):
        for s in dyn_rwd_model.keys():
            if s not in Vfunc:
                Vfunc[s] = 0.0
            max_nV = -100.0
            for a in dyn_rwd_model[s].keys():
                totalV = 0.0
                for sn in dyn_rwd_model[s][a].keys():
                    if sn not in Vfunc:
                        Vfunc[sn] = 0.0
                    totalV += dyn_rwd_model[s][a][sn][0] * (dyn_rwd_model[s][a][sn][1] + gamma * Vfunc[sn])
                if totalV > max_nV:
                    max_nV = totalV
            Vfunc[s] = max_nV
    policy = {}
    for s in dyn_rwd_model.keys():
        if s not in policy:
            policy[s] = 0.0
        max_nV = -100.0
        best_a = None
        for a in dyn_rwd_model[s].keys():
            totalV = 0.0
            for sn in dyn_rwd_model[s][a].keys():
                totalV += dyn_rwd_model[s][a][sn][0] * (dyn_rwd_model[s][a][sn][1] + gamma * Vfunc[sn])
            if totalV > max_nV:
                max_nV = totalV
                best_a = a
        policy[s] = best_a
    return policy

def main():
    path = 'data/value_iter_swingup_discrete'
    logger.reset()
    logger.configure(path)

    env = gym.make('DartCartPole-v1')
    obs_disc = bin_disc([[50, 1, -1], [50, 0.5, -0.5], [50, 10, -10], [50, 20, -20]])
    act_disc = bin_disc([[50, -1.0, 1]])
    obs_disc_dim = 50*50*50*50
    act_disc_dim = 10

    dyn_model = learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim)
    policy = optimize_policy(dyn_model, 0.96)

    joblib.dump(dyn_model, path+'/dyn_model.pkl', compress=True)
    joblib.dump(policy, path + '/policy.pkl', compress=True)



if __name__ == '__main__':
    main()





















