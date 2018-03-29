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
from baselines.valueiteration.utils import *
from baselines.ppo1 import pposgd_simple
from baselines.valueiteration.mlp_net import *

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
    from baselines.valueiteration import mlp_additive_policy, pposgd_disc
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_additive_policy.MlpAddPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3, gmm_comp=1, learn_additive=False)

    env.env.use_disc_ref_policy = True
    env.env.learn_additive_pol = True
    path = 'data/value_iter_cartpole_discrete_adaptsampled_fromtrained'
    disc_pi = joblib.load(path + '/policy.pkl')
    ref_policy_funcs = joblib.load(
        path + '/ref_policy_funcs.pkl')
    env.env.disc_funcs = ref_policy_funcs
    env.env.disc_policy = disc_pi

    fitpol_params = joblib.load(path+'/fitpolparams.pkl')
    fitted_policy = MlpNet('fitpol', ref_policy_funcs[1].ndim, ref_policy_funcs[2].ndim, hid_size=64, num_hid_layers=3)
    env.env.disc_fit_policy = fitted_policy

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed + MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)
    pposgd_disc.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(2000),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='constant',
                        callback=callback,
                        fitted_policy_params = [fitted_policy, fitpol_params],
                        #init_policy_params = joblib.load('data/ppo_DartHopper-v10_using_no_disc_ref_policy/policy_params_40.pkl')
        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_discfitpolicy_additive')
    train(args.env, num_timesteps=int(5000*200), seed=args.seed)


if __name__ == '__main__':
    main()
