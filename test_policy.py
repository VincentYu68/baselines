__author__ = 'yuwenhao'

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
from gym import wrappers
import tensorflow as tf
from baselines.ppo1 import mlp_policy, pposgd_simple
import baselines.common.tf_util as U

np.random.seed(1)

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=3)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3dRestricted-v1')

    if hasattr(env.env, 'disableViewer'):
        env.env.disableViewer = False
        '''if hasattr(env.env, 'resample_MP'):
        env.env.resample_MP = False'''

    record = False
    if len(sys.argv) > 3:
        record = int(sys.argv[3]) == 1
    if record:
        env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True)
    else:
        env_wrapper = env

    sess = tf.InteractiveSession()

    policy = None
    if len(sys.argv) > 2:
        policy_params = joblib.load(sys.argv[2])
        ob_space = env.observation_space
        ac_space = env.action_space
        policy = policy_fn("pi", ob_space, ac_space)

        U.initialize()

        cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
        orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]
        for i in range(len(policy.get_variables())):
            assign_op = policy.get_variables()[i].assign(
                policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            sess.run(assign_op)

        if 'curriculum' in sys.argv[2] and 'policy_params.pkl' in sys.argv[2]:
            if os.path.isfile(sys.argv[2].replace('policy_params.pkl', 'init_poses.pkl')):
                init_qs, init_dqs = joblib.load(sys.argv[2].replace('policy_params.pkl', 'init_poses.pkl'))
                env.env.init_qs = init_qs
                env.env.init_dqs = init_dqs

    print('===================')

    o = env_wrapper.reset()

    rew = 0

    actions = []

    traj = 1
    ct = 0
    vel_rew = []
    action_pen = []
    deviation_pen = []
    rew_seq = []
    com_z = []
    x_vel = []
    foot_contacts = []
    d=False

    while ct < traj:
        if policy is not None:
            ac, vpred = policy.act(False, o)
            act = ac
        else:
            act = env.action_space.sample()
        actions.append(act)
        o, r, d, env_info = env_wrapper.step(act)

        if 'action_pen' in env_info:
            action_pen.append(env_info['action_pen'])
        if 'vel_rew' in env_info:
            vel_rew.append(env_info['vel_rew'])
        rew_seq.append(r)
        if 'deviation_pen' in env_info:
            deviation_pen.append(env_info['deviation_pen'])

        com_z.append(o[1])
        foot_contacts.append(o[-2:])

        rew += r

        env_wrapper.render()

        #time.sleep(0.1)
        if len(o) > 25:
            x_vel.append(env.env.robot_skeleton.dq[0])

        if len(foot_contacts) > 400:
            if np.random.random() < 0.03:
                print('q ', np.array2string(env.env.robot_skeleton.q, separator=','))
                print('dq ', np.array2string(env.env.robot_skeleton.dq, separator=','))

        if d:
            ct += 1
            print('reward: ', rew)
            o=env_wrapper.reset()
            #break
    print('avg rew ', rew / traj)

    if sys.argv[1] == 'DartWalker3d-v1':
        rendergroup = [[0,1,2], [3,4,5, 9,10,11], [6,12], [7,8, 12,13]]
        for rg in rendergroup:
            plt.figure()
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHumanWalker-v1':
        rendergroup = [[0,1,2, 6,7,8], [3,9], [4,5,10,11], [12,13,14], [15,16,7,18]]
        titles = ['thigh', 'knee', 'foot', 'waist', 'arm']
        for i,rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    plt.figure()
    plt.title('rewards')
    plt.plot(rew_seq, label='total rew')
    plt.plot(action_pen, label='action pen')
    plt.plot(vel_rew, label='vel rew')
    plt.plot(deviation_pen, label='dev pen')
    plt.legend()
    plt.figure()
    plt.title('com z')
    plt.plot(com_z)
    plt.figure()
    plt.title('x vel')
    plt.plot(x_vel)
    foot_contacts = np.array(foot_contacts)
    plt.figure()
    plt.title('foot contacts')
    plt.plot(1-foot_contacts[:, 0])
    plt.plot(1-foot_contacts[:, 1])
    plt.show()



