#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines import logger
from baselines import bench
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import gym, logging
import os.path as osp

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=2000, max_kl=0.01, cg_iters=10, cg_damping=1e-5,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=15, vf_stepsize=4e-4)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    logger.reset()
    logger.configure('trpo8000_lowdim_' + args.env + str(args.seed))
    train(args.env, num_timesteps=8000000, seed=args.seed)


if __name__ == '__main__':
    main()
