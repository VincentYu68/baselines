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



def learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim, policy = None, disc_policy = True, collected_data = []):
    dyn_rwd_model = {}
    collected_steps = 0

    all_state = []
    while collected_steps < 45000:
        d = False
        env.reset()
        while not d:
            bg_step = state_filter_hopper(env.env.state_vector())
            if policy is None:
                act = env.action_space.sample()
            else:
                if disc_policy:
                    if np.random.random() < 0.95 and obs_disc(bg_step) in policy and policy[obs_disc(bg_step)] is not None:
                        act = act_disc.samp_state(policy[obs_disc(bg_step)])
                    else:
                        act = env.action_space.sample()
                else:
                    if np.random.random() < 0.99:
                        act = policy.act(True, env.env._get_obs())[0]
                    else:
                        act = env.action_space.sample()
            prev_act = act_disc(act)
            bg_step_disc = obs_disc(bg_step)

            o,r,d,_ = env.step(act)
            ed_step = state_filter_hopper(env.env.state_vector())
            collected_data.append([bg_step, act, ed_step, r])
            collected_steps += 1
            all_state.append(bg_step)

            ed_step_disc = obs_disc(ed_step)
            if d:
                print(collected_steps)

    max_vals = np.max(all_state, axis=0)
    min_vals = np.min(all_state, axis=0)
    print(np.max(all_state, axis=0), np.min(all_state, axis=0), len(collected_data))
    for d in range(obs_disc.ndim):
        obs_disc.disc_scheme[d][1] = max_vals[d] + 0.0001
        obs_disc.disc_scheme[d][2] = min_vals[d]

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
    #print(dyn_rwd_model)
    return dyn_rwd_model, collected_data, obs_disc

def optimize_policy(dyn_rwd_model, gamma, Vfunc = {}):
    for iter in range(500):
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
    return Vfunc, policy

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3, gmm_comp=1)

def main():
    path = 'data/value_iter_hopper_discrete'
    logger.reset()
    logger.configure(path)

    env = gym.make('DartHopper-v1')
    '''obs_disc = bin_disc([[50, 0, -2.0], [50, 2*np.pi, 0.0], [50, 1.0, -1], [50, 3, -3]])'''
    obs_disc = bin_disc([[20, 0.6, -0.5], [10, 0.2, -0.2], [50, 0.4, -2.62], [50, 0.4, -2.62], [20, 1.0, -1.0],
            [50, 10.0, -1.0], [10, 10.0, -10.0], [10, 10.0, -10.0], [10, 10.0, -10.0], [10, 10.0, -10.0], [10, 10.0, -10.0]])
    act_disc = bin_disc([[5, 1.0, -1.0], [5, 1.0, -1.0], [5, 1.0, -1.0]])
    obs_disc_dim = 1
    act_disc_dim = 1
    for s in obs_disc.disc_scheme:
        obs_disc_dim *= s[0]
    for s in act_disc.disc_scheme:
        act_disc_dim *= s[0]

    policy = None
    sess = tf.InteractiveSession()
    policy_params = joblib.load(
        'data/ppo_DartHopper-v10_using_no_disc_ref_policy/policy_params_40.pkl')
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

    dyn_model, col_data, obs_disc = learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim, policy=policy, disc_policy = False)
    Vfunc, policy = optimize_policy(dyn_model, 0.99)

    for iter in range(0):
        dyn_model, col_data, obs_disc = learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim, policy = policy, collected_data=col_data)
        Vfunc, policy = optimize_policy(dyn_model, 0.99, Vfunc = Vfunc)
        joblib.dump(dyn_model, path+'/dyn_model_'+str(iter)+'.pkl', compress=True)
        joblib.dump(policy, path + '/policy_'+str(iter)+'.pkl', compress=True)
    joblib.dump(dyn_model, path + '/dyn_model.pkl', compress=True)
    joblib.dump(policy, path + '/policy.pkl', compress=True)
    joblib.dump([Vfunc, obs_disc, act_disc], path + '/ref_policy_funcs.pkl', compress=True)

if __name__ == '__main__':
    main()





















