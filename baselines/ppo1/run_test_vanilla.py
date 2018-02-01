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
from baselines.valueiteration.value_iteration_learn import *

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
    from baselines.ppo1 import mlp_mirror_policy
    from baselines.valueiteration import pposgd_disc
    U.make_session(num_cpu=1).__enter__()
    env = gym.make(env_id)

    '''path = 'data/value_iter_truehopper_discrete'
    [Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn] = joblib.load(path + '/ref_policy_funcs.pkl')
    env.env.use_disc_ref_policy = True
    env.env.disc_ref_weight = 0.01
    env.env.disc_funcs = [Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn]'''

    def policy_fn(name, ob_space, ac_space):
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=64, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array(
                                                     [1]*2),
                                                 action_permutation=np.array(
                                                     [0.001]*1))
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    gym.logger.setLevel(logging.WARN)

    '''s_disc = []
    for i in range(11):
        s_disc.append([30, 0.0, -0.0])
    obs_disc = bin_disc(s_disc)
    act_disc = bin_disc([[10, 1.01, -1.01], [10, 1.01, -1.01], [10, 1.01, -1.01]])
    state_filter_fn = state_filter_hopper
    state_unfilter_fn = state_unfilter_hopper'''

    obs_disc = bin_disc([[51, 0, -0.01], [51, 0.0, -0.01], [51, 0.0, -0.01], [51, 0.0, -0.01]])
    act_disc = bin_disc([[100, 1.01, -1.01]])
    state_filter_fn = state_filter_cartpole
    state_unfilter_fn = state_unfilter_cartpole

    pposgd_disc.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(500),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
                        callback=callback,
                      sym_loss_weight = 0.0,
                      #ref_policy_params=joblib.load('data/ppo_DartCartPoleSwingUp-v11_vanilla/policy_params.pkl')
                      #discrete_learning = [obs_disc, act_disc, state_filter_fn, state_unfilter_fn, 0.2],
                        #init_policy_params=joblib.load('data/ppo_DartHopper-v12_vanilla/policy_params.pkl')
        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalker-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_vf_vanilla_weak_2k')
    #logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy05_bal_vel4smooth_mirror_up1fwd01ltl1_spinepen1yaw001_thighyawpen005_initbentelbow_velrew3_dcontrolconstraint1_strongerarm_asinput_treadmill')
    train(args.env, num_timesteps=int(500*4*100), seed=args.seed)

if __name__ == '__main__':
    main()
