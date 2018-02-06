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

def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    policies = localv['pis']
    task_size = localv['task_size']
    for t in range(task_size):
        save_dict = {}
        variables = policies[t].get_variables()
        for i in range(len(variables)):
            cur_val = variables[i].eval()
            save_dict[variables[i].name] = cur_val
        joblib.dump(save_dict, logger.get_dir()+'/policy_params_t'+str(t)+'_'+str(localv['iters_so_far'])+'.pkl', compress=True)
        joblib.dump(save_dict, logger.get_dir() + '/policy_params_t'+str(t) + '.pkl', compress=True)


def train(env_id, num_timesteps, batch, seed, split_iter, split_percent):
    from baselines.split_net import mlp_split_policy, pposgd_split
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_split_policy.MlpSplitPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3, gmm_comp=1)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed+MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)
    pposgd_split.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(batch),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
                        callback=callback,
                       split_iter=split_iter,
                       split_percent=split_percent,
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--split_iter', help='iteration number that starts splitting', type=int, default=0)
    parser.add_argument('--split_percent', help='number of splitted parameters', type=float, default=0.0)
    parser.add_argument('--batch', help='batch per thread', type=int, default=1000)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_split_'+str(args.split_iter)+'_'+str(args.split_percent)+'_2task_forwardbackward_'+str(args.batch))
    train(args.env, num_timesteps=int(2000*4*300), batch = args.batch, seed=args.seed, split_iter=args.split_iter, split_percent=args.split_percent)

if __name__ == '__main__':
    main()