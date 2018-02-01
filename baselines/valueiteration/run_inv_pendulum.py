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
from baselines.ppo1 import mlp_policy
from baselines.valueiteration.utils import *
from baselines.valueiteration.value_iteration_learn import *

set_global_seeds(1)

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3, gmm_comp=1)

def main():

    path = 'data/value_iter_cartpole_discrete_v4'
    logger.reset()
    logger.configure(path)

    env = gym.make('DartCartPoleSwingUp-v1')
    env.seed(0)
    #obs_disc = bin_disc([[50, 0, -0.01], [50, 0.0, -0.01]])
    #act_disc = bin_disc([[10, 1.01, -1.01]])
    obs_disc = bin_disc([[50, 0, -0.01], [50, 0.0, -0.01], [50, 0.0, -0.01], [50, 0.0, -0.01]])
    act_disc = bin_disc([[50, 1.01, -1.01]])

    '''s_disc = []
    for i in range(11):
        s_disc.append([30, 0.0, -0.0])
    obs_disc = bin_disc(s_disc)
    act_disc = bin_disc([[10, 1.01, -1.01], [10, 1.01, -1.01], [10, 1.01, -1.01]])
    #obs_disc = bin_disc([[5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1], [5, -1, 1]])
    #act_disc = bin_disc([[4, 1.0, -1.0], [4, 1.0, -1.0], [4, 1.0, -1.0], [4, 1.0, -1.0], [4, 1.0, -1.0]])'''

    obs_disc_dim = 1
    act_disc_dim = 1
    for s in obs_disc.disc_scheme:
        obs_disc_dim *= s[0]
    for s in act_disc.disc_scheme:
        act_disc_dim *= s[0]

    state_filter_fn = state_filter_cartpole
    state_unfilter_fn = state_unfilter_cartpole

    policy = None
    '''sess = tf.InteractiveSession()
    policy_params = joblib.load(
        'data/ppo_DartCartPoleSwingUp-v11_vanilla/policy_params.pkl')
    ob_space = env.observation_space
    ac_space = env.action_space
    policy = policy_fn("pi", ob_space, ac_space)
    U.initialize()
    cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]
    vars = policy.get_variables()
    for i in range(len(policy.get_variables())):
        assign_op = policy.get_variables()[i].assign(
            policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)
    env.env.use_disc_ref_policy = None'''

    dyn_model, col_data, obs_disc = learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim, state_filter_fn, state_unfilter_fn, policy=policy, disc_policy = False)
    Vfunc, policy = optimize_policy(dyn_model, 0.99)

    for iter in range(50):
        print('--------------- Iteration ', str(iter), ' -------------------')
        dyn_model, col_data, obs_disc = learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim, state_filter_fn, state_unfilter_fn, policy = policy, collected_data=col_data)
        Vfunc, policy = optimize_policy(dyn_model, 0.99, Vfunc = Vfunc)
        joblib.dump(dyn_model, path+'/dyn_model_'+str(iter)+'.pkl', compress=True)
        joblib.dump(policy, path + '/policy_'+str(iter)+'.pkl', compress=True)
        joblib.dump([Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn], path + '/ref_policy_funcs_'+str(iter)+'.pkl', compress=True)

        joblib.dump(dyn_model, path + '/dyn_model.pkl', compress=True)
        joblib.dump(policy, path + '/policy.pkl', compress=True)
        joblib.dump([Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn], path + '/ref_policy_funcs.pkl', compress=True)
    joblib.dump(dyn_model, path + '/dyn_model.pkl', compress=True)
    joblib.dump(policy, path + '/policy.pkl', compress=True)
    joblib.dump([Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn], path + '/ref_policy_funcs.pkl', compress=True)

if __name__ == '__main__':
    main()





















