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
                                                     [0.0001, -1, 2, -3, -4, 9, 10, 11, 12, 5, 6, 7, 8, 17, 18, 19, 20,
                                                      13,
                                                      14, 15, 16,
                                                      21, 22, -23, 24, -25, -26, 31, 32, 33, 34, 27, 28, 29, 30, 39, 40,
                                                      41,
                                                      42, 35, 36, 37, 38, 44, 43, 46, 45, 47]),
                                                 action_permutation=np.array(
                                                     [4, 5, 6, 7, 0.0001, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]),
                                                 )
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed+MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)
    pposgd_mirror.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(2500),
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            callback=callback,
            sym_loss_weight=2.0,
            positive_rew_enforce=False,
            #init_policy_params = joblib.load('data/ppo_DartHumanWalker-v12_energy01armlowweight_vel15_mirror_up03fwd01ltl22_spinepen1yaw001_thighyawpen005_initbentelbow_velrew3_dcon1_asinput_damping2kneethigh_thigh250knee150_treadmill/policy_params.pkl'),
            reward_drop_bound=True,
            #init_policy_params = joblib.load('data/ppo_DartHumanWalker-v1124_energy25_vel3_kd1000_mirror_up1fwd01ltl15_spinepen1yaw001_thighyawpen005_initbentelbow_runningavg3_dcontrolconstraint1_asinput_damping2_fromvel3_kd500/policy_params.pkl')
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartDogRobot-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy2_vel2_1s_mirror_velrew3_dcon1_asinput_damping5_oldinitjoint_newnewstrength15x_05fwdpen_01velpentransition_ab3_matchedseed')
    #logger.configure('data/ppo_'+args.env+str(args.seed)+'_energy05_bal_vel4smooth_mirror_up1fwd01ltl1_spinepen1yaw001_thighyawpen005_initbentelbow_velrew3_dcontrolconstraint1_strongerarm_asinput_treadmill')
    train_mirror(args.env, num_timesteps=int(5000*4*3000), seed=args.seed)


if __name__ == '__main__':
    main()




