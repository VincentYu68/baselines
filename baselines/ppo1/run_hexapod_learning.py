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
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    joblib.dump(save_dict, logger.get_dir()+'/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)

def train_mirror(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_mirror_policy, pposgd_mirror
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=64, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array(
                                                     [0.0001,-1,2,-3,-4, 8,9,10, 5,6,7, 14,15,16, 11,12,13, 20,21,22,17,18,19,
                                                      23,24,-25,26,-27,-28, 32,33,34, 29,30,31,  38,39,40,35,36,37, 44,45,46,41,42,43,
                                                      48,47, 50,49, 52,51, 53]),
                                                 action_permutation=np.array(
                                                     [3,4,5,0.0001,1,2, 9,10,11,6,7,8, 15,16,17,12,13,14]))
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed+MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)
    pposgd_mirror.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(1500),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            callback=callback,
            sym_loss_weight=2.0,
            positive_rew_enforce=False,
            #init_policy_params = joblib.load('data/ppo_DartHumanWalker-v1156_energy1_vel55_mirror_up1fwd01ltl15_spinepen1yaw001_thighyawpen005_initbentelbow_runningavg4_dcontrolconstraint1_asinput_damping2kneethigh_thigh250knee60/policy_params.pkl'),
            reward_drop_bound=True,
            #init_policy_params = joblib.load('data/ppo_DartHumanWalker-v1124_energy25_vel3_kd1000_mirror_up1fwd01ltl15_spinepen1yaw001_thighyawpen005_initbentelbow_runningavg3_dcontrolconstraint1_asinput_damping2_fromvel3_kd500/policy_params.pkl')
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHexapod-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy005_vel8_mirror_velrew3_asinput')
    train_mirror(args.env, num_timesteps=int(5000*4*800), seed=args.seed)


if __name__ == '__main__':
    main()
