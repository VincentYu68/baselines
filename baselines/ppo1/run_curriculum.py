#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

from baselines.ppo1 import mlp_mirror_policy, pposgd_mirror

def evaluate_policy(env, policy, reps=20):
    avg_return = 0.0
    for i in range(reps):  # average performance over 10 trajectories
        o = env.reset()
        while True:
            o, rew, done, _ = env.step(policy.act(True, o)[0])
            avg_return += rew
            if done:
                break
    return avg_return / reps


def binary_search_curriculum(env, policy, anchor, direction, threshold, max_step):
    current_min = 0.0
    if anchor[0] / np.linalg.norm(anchor) < np.abs(direction[0]):
        current_max = np.abs(anchor[0] / direction[0])
    else:
        current_max = np.abs(anchor[1] / direction[1])
    bound_point = anchor + direction * current_max
    env.env.env.anchor_kp=bound_point
    bound_performance = evaluate_policy(env, policy)
    if (bound_performance - threshold) < np.abs(threshold * 0.1) and bound_performance > threshold:
        return bound_point, bound_performance

    for i in range(max_step):
        current_step = 0.5 * (current_max + current_min)
        current_point = anchor + current_step * direction
        env.env.env.anchor_kp=current_point
        curr_perf = evaluate_policy(env, policy)
        if (curr_perf - threshold) < np.abs(threshold * 0.1) and curr_perf > threshold:
            return current_point, curr_perf
        if curr_perf > threshold:
            current_min = current_step
        if curr_perf < threshold:
            current_max = current_step
    return anchor + current_min * direction, curr_perf

def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    joblib.dump(save_dict, logger.get_dir()+'/policy_params_'+str(localv['env'].env.env.anchor_kp)+'_'+str(localv['iters_so_far'])+'.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalker-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--init_policy', help='Initial Policy', default='data/ppo_curriculum_DartHumanWalker-v1_0_0.8_0.5_2500/policy_params.pkl')
    parser.add_argument('--init_curriculum', help='Initial Curriculum', nargs='+', default=[0, 0])
    parser.add_argument('--ref_policy', help='Reference Policy', default='data/ppo_DartHumanWalker-v13_4energy_vel_bal_mirror/policy_params.pkl')
    parser.add_argument('--ref_curriculum', help='Reference Curriculum', nargs='+', default=[6000, 3000])
    parser.add_argument('--anc_thres', help='Anchor Threshold', type=float, default=0.8)
    parser.add_argument('--prog_thres', help='Progress Threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=2500)
    parser.add_argument('--max_iter', help='Maximum Iteration', type=int, default=2000)
    args = parser.parse_args()
    logger.reset()
    logger.configure('data/ppo_curriculum_cont_'+args.env+'_'+str(args.seed)+'_'+str(args.anc_thres)+'_'+str(args.prog_thres)+'_'+str(args.batch_size))

    sess = U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env)

    ob_space = env.observation_space
    ac_space = env.action_space

    def policy_fn(name, ob_space, ac_space):
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=64, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array(
                                                     [0.0001, -1, 2, -3, -4, -11, 12, -13, 14, 15, 16, -5, 6, -7, 8, 9,
                                                      10, -17, 18, -19, -24, 25, -26, 27, -20, 21, -22, 23,
                                                      28, 29, -30, 31, -32, -33, -40, 41, -42, 43, 44, 45, -34, 35, -36,
                                                      37, 38, 39, -46, 47, -48, -53, 54, -55, 56, -49, 50, -51, 52, 58,
                                                      57]),
                                                 action_permutation=np.array(
                                                     [-6, 7, -8, 9, 10, 11, -0.001, 1, -2, 3, 4, 5, -12, 13, -14, -19,
                                                      20, -21, 22, -15, 16, -17, 18]))

    policy = policy_fn('policy', ob_space, ac_space)
    init_curriculum = np.array(args.init_curriculum)
    ref_policy = policy_fn('ref_policy', ob_space, ac_space)
    ref_curriculum = np.array(args.ref_curriculum)

    policy_params = joblib.load(args.init_policy)
    ref_policy_params = joblib.load(args.ref_policy)
    U.initialize()
    cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]
    ref_scope = list(ref_policy_params.keys())[0][0:list(ref_policy_params.keys())[0].find('/')]
    print(cur_scope, orig_scope)
    for i in range(len(policy.get_variables())):
        assign_op = policy.get_variables()[i].assign(
            policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)
        assign_op = ref_policy.get_variables()[i].assign(
            ref_policy_params[ref_policy.get_variables()[i].name.replace('ref_'+cur_scope, ref_scope, 1)])
        sess.run(assign_op)


    anchor_threshold = args.anc_thres
    progress_threshold = args.prog_thres

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)

    curriculum_evolution = []

    env.env.env.anchor_kp = ref_curriculum
    ref_score = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        ref_score = evaluate_policy(env, ref_policy, 20)
    ref_score=MPI.COMM_WORLD.bcast(ref_score, root = 0)
    reference_score = ref_score * progress_threshold
    reference_anchor_score = ref_score * anchor_threshold
    env.env.env.anchor_kp = init_curriculum

    current_curriculum = np.copy(init_curriculum)
    print('reference scores: ', reference_score, reference_anchor_score)

    previous_params = policy_params
    for iter in range(args.max_iter):
        print('curriculum iter ', iter)
        print('ref score: ', reference_anchor_score)
        opt_pi = pposgd_mirror.learn(env, policy_fn,
                                    max_timesteps=args.batch_size * 200,
                                    timesteps_per_batch=int(args.batch_size),
                                    clip_param=0.2, entcoeff=0.0,
                                    optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                    gamma=0.99, lam=0.95, schedule='linear',
                                    callback=callback,
                                    sym_loss_weight=2.0,
                                    return_threshold=reference_anchor_score,
                                    init_policy_params = previous_params,
                                    policy_scope='pi'+str(iter),
                                    )
        if np.linalg.norm(current_curriculum) >= 0.0001:
            closest_candidate = None
            if MPI.COMM_WORLD.Get_rank() == 0:
                directions = [np.array([-1, 0]), np.array([0, -1]), -current_curriculum / np.linalg.norm(current_curriculum)]
                int_d1 = directions[0] + directions[2]
                int_d2 = directions[1] + directions[2]
                directions.append(int_d1 / np.linalg.norm(int_d1))
                directions.append(int_d2 / np.linalg.norm(int_d2))
                candidate_next_anchors = []
                for direction in directions:
                    found_point, perf = binary_search_curriculum(env, opt_pi, current_curriculum, direction, reference_score, 6)
                    print(direction, found_point, perf)
                    candidate_next_anchors.append(found_point)
                    if closest_candidate is None:
                        closest_candidate = np.copy(found_point)
                    elif np.linalg.norm(closest_candidate) > np.linalg.norm(found_point):
                        closest_candidate = np.copy(found_point)
                if np.linalg.norm(closest_candidate) < 0.5:
                    closest_candidate = np.array([0, 0])
            closest_candidate = MPI.COMM_WORLD.bcast(closest_candidate, root=0)

            current_curriculum = np.copy(closest_candidate)
        env.env.env.anchor_kp = current_curriculum
        curriculum_evolution.append(current_curriculum)
        print('Current curriculum: ', current_curriculum)
        opt_variable = opt_pi.get_variables()
        previous_params = {}
        for i in range(len(opt_variable)):
            cur_val = opt_variable[i].eval()
            previous_params[opt_variable[i].name] = cur_val
        if np.linalg.norm(current_curriculum) < 0.0001:
            if reference_anchor_score < ref_score:
                reference_anchor_score = ref_score
            else:
                break

    env.close()


if __name__ == '__main__':
    main()
