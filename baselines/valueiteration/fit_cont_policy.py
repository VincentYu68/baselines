#!/usr/bin/env python
from baselines.common import Dataset, set_global_seeds, tf_util as U
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
from baselines.common.mpi_adam import MpiAdam
#import matplotlib.pyplot as plt

# fit continuous policy from a discrete policy
def main():
    U.make_session(num_cpu=1).__enter__()

    path = 'data/value_iter_cartpole_discrete_adaptsampled_fromtrained'

    dyn_model = joblib.load(path + '/dyn_model.pkl')
    policy = joblib.load(path + '/policy.pkl')
    [Vfunc, obs_disc, act_disc, state_filter_fn, state_unfilter_fn] = joblib.load(path + '/ref_policy_funcs.pkl')

    optim_epochs = 150
    optim_batchsize = 64
    optim_stepsize = 1e-3

    print('model loading done')
    print(obs_disc.disc_scheme)

    obs_dim = obs_disc.ndim
    act_dim = act_disc.ndim

    qinput = U.get_placeholder(name="qinput", dtype=tf.float32, shape=[None] + [obs_dim+act_dim, ])
    fitted_qfunc = MlpNet('fitq', obs_dim+act_dim, 1, hid_size=128, num_hid_layers=3, input_placeholder=qinput)
    qvarlist = fitted_qfunc.get_trainable_variables()
    qoptimizer = MpiAdam(qvarlist)

    qtarget = U.get_placeholder(name="qtarget", dtype=tf.float32, shape=[None] + [1,])

    qloss = U.mean(tf.square(qtarget - fitted_qfunc.mean))
    qlossngrad = U.function([qinput, qtarget], [qloss, U.flatgrad(qloss, qvarlist)])

    ob = U.get_placeholder(name="polinput", dtype=tf.float32, shape=[None] + [obs_dim, ])
    fitted_policy = MlpNet('fitpol', obs_dim, act_dim, hid_size=64, num_hid_layers=3, input_placeholder=ob)
    var_list = fitted_policy.get_trainable_variables()
    optimizer = MpiAdam(var_list)

    qpred = fitted_qfunc.get_symbolic_output(U.concatenate([ob, fitted_policy.mean], axis=1))
    loss = -U.mean(qpred)
    lossngrad = U.function([ob], [loss, U.flatgrad(loss, var_list)])

    U.initialize()

    # Collect training data for q function
    qinputs = []
    qtargets = []
    observations = []
    for i in range(1):
        for s in dyn_model:
            observations.append(obs_disc.samp_state(s))
            for a in dyn_model[s]:
                o = obs_disc.samp_state(s)
                ca = act_disc.samp_state(a)
                qinputs.append(np.concatenate([o, ca]))
                avg_q = 0
                for sn in dyn_model[s][a].keys():
                    avg_q += 0.99*dyn_model[s][a][sn][0]*Vfunc[sn] + dyn_model[s][a][sn][1]
                qtargets.append([avg_q])

    qtargets = np.array(qtargets)
    qtargets = (qtargets - np.mean(qtargets)) / np.std(qtargets)

    q_dataset = Dataset(dict(qinputs=np.array(qinputs), qtargets=np.array(qtargets)), shuffle=True)

    qlosses = []
    for _ in range(optim_epochs):
        losses_one_epoch = []
        for batch in q_dataset.iterate_once(optim_batchsize):
            newloss, g = qlossngrad(batch["qinputs"], batch["qtargets"])
            qoptimizer.update(g, optim_stepsize)
            losses_one_epoch.append(newloss)
        qlosses.append(np.mean(losses_one_epoch))
        print('Epoch ', _, ' :', np.mean(losses_one_epoch))

    errors = []
    for obid in range(len(qinputs)):
        error = np.abs(fitted_qfunc.pred(qinputs[obid]) - qtargets[obid])
        errors.append(error[0][0])
    errors = np.array(errors)
    sorted_index = np.argsort(-errors)
    top_error_id = sorted_index[0:16]
    for eid in top_error_id:
        print(qinputs[eid], fitted_qfunc.pred(qinputs[eid]), qtargets[eid], errors[eid])

    # fit policy
    print('Fit policy')
    print('Collected ', len(observations))
    dataset = Dataset(dict(ob=np.array(observations)), shuffle=True)

    losses = []
    for _ in range(optim_epochs):
        losses_one_epoch = []
        for batch in dataset.iterate_once(optim_batchsize):
            newloss, g = lossngrad(batch["ob"])
            optimizer.update(g, optim_stepsize)
            losses_one_epoch.append(newloss)
        losses.append(np.mean(losses_one_epoch))
        print('Epoch ', _, ' :', np.mean(losses_one_epoch))

    fitpolparams = {}
    for p in fitted_policy.get_variables():
        fitpolparams[p.name] = p.eval()
    joblib.dump(fitpolparams, path+'/fitpolparams.pkl', compress=True)

    #print(losses)

    '''errors = []
    for obid in range(len(observations)):
        error = np.abs(fitted_policy.pred(observations[obid]) - acttargets[obid])
        errors.append(error[0][0])
    errors = np.array(errors)
    sorted_index = np.argsort(-errors)
    top_error_id = sorted_index[0:16]
    for eid in top_error_id:
        print(observations[eid], fitted_policy.pred(observations[eid]), acttargets[eid], errors[eid])'''

    #plt.figure()
    #plt.plot(losses)
    #plt.savefig(path+'/polfit_learning_curve.jpg')'''




if __name__ == '__main__':
    main()
