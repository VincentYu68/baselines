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

def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    joblib.dump(save_dict, logger.get_dir()+'/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(5000),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
                        callback=callback,
        )
    env.close()

def train_mirror(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_mirror
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_mirror.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(5000),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            callback=callback,
            observation_permutation=np.array(
                [0.0001, -1, 2, -3, -4, -5, -6, 7, 14, -15, -16, 17, 18, -19, 8, -9, -10, 11, 12, -13,
                 20, 21, -22, 23, -24, -25, -26, -27, 28, 35, -36, -37, 38, 39, -40, 29, -30, -31, 32, 33,
                 -34, 42, 41]),
            action_permutation=np.array([-0.0001, -1, 2, 9, -10, -11, 12, 13, -14, 3, -4, -5, 6, 7, -8]),
            sym_loss_weight=1.0
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_03energypen_3d')
    train(args.env, num_timesteps=75000000, seed=args.seed)


if __name__ == '__main__':
    main()
