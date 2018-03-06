from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import copy


def traj_segment_generator(pi, env, horizon, stochastic, task_id):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    env.env.env.enforce_task_id = task_id
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float64')
    vpreds = np.zeros(horizon, 'float64')
    news = np.zeros(horizon, 'int32')
    task_ids = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "task_ids": task_ids}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, envinfo = env.step(ac)
        if 'task_id' in envinfo:
            task_ids[i] = envinfo['task_id']
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            broke = False
            if 'broke_sim' in envinfo:
                if envinfo['broke_sim']:
                    broke = True
            if not broke:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
            else:
                t -= (cur_ep_len + 1)
            cur_ep_ret = 0
            cur_ep_len = 0
            env.env.env.enforce_task_id = task_id
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float64')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def optimize_one_iter(datasets, lossandgrads, task_adams, optim_epochs, optim_stepsize, task_size, optim_batchsize,
                      cur_lrmult, share_params_list):
    learning_curve = []
    for _ in range(optim_epochs):
        losses = []  # list of tuples, each of which gives the loss for a minibatch
        while datasets[0]._next_id <= datasets[0].n - optim_batchsize:
            avg_losses = []
            for t in range(task_size):
                batch = datasets[t].next_batch(optim_batchsize)
                *newlosses, g = lossandgrads[t](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                task_adams[t].update(g, optim_stepsize * cur_lrmult)
                # synchronize the shared parameters
                for t2 in range(task_size):
                    if t == t2 or np.sum(share_params_list[t2, :] == share_params_list[t, :]) == 0:
                        continue
                    gmask = share_params_list[t2, :] == share_params_list[t, :]
                    cur_param = task_adams[t2].getflat()
                    cur_param[gmask] = task_adams[t].getflat()[gmask]
                    task_adams[t2].setfromflat(cur_param)
                    task_adams[t2].m[gmask] = task_adams[t].m[gmask]
                    task_adams[t2].v[gmask] = task_adams[t].v[gmask]
                avg_losses.append(newlosses)
            losses.append(np.mean(avg_losses, axis=0))

        for t in range(task_size):
            datasets[t]._next_id = 0
            datasets[t].shuffle()
        learning_curve.append(np.mean(losses, axis=0))
    return learning_curve


def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          return_threshold=None,  # termiante learning if reaches return_threshold
          op_after_init=None,
          init_policy_params=None,
          policy_scope=None,
          max_threshold=None,
          positive_rew_enforce=False,
          reward_drop_bound=300,
          min_iters=0,
          ref_policy_params=None,
          rollout_length_thershold=None,
          split_iter=0,
          split_percent=0.0,
          split_interval=1000000,
          adapt_split=False,
          rand_split=False
          ):
    # Setup losses and stuff
    # ----------------------------------------
    task_size = env.env.env.tasks.task_num

    ob_space = env.observation_space
    ac_space = env.action_space
    pis = []
    oldpis = []
    if policy_scope is None:
        for i in range(task_size):
            pis.append(policy_func("pi" + str(i), ob_space, ac_space))  # Construct network for new policy
            oldpis.append(policy_func("oldpi" + str(i), ob_space, ac_space))  # Network for old policy
    else:
        for i in range(task_size):
            pis.append(policy_func(policy_scope, ob_space, ac_space))  # Construct network for new policy
            oldpis.append(policy_func("old" + policy_scope, ob_space, ac_space))

    atarg = tf.placeholder(dtype=tf.float64, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float64, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float64,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pis[0].pdtype.sample_placeholder([None])

    loss_names = ["pol_surr", "vf_loss"]
    task_losses = []
    task_assign_old_eq_new = []
    task_compute_losses = []
    task_adams = []
    lossandgrads = []
    get_flats = []
    set_from_flats = []
    for t in range(task_size):
        ratio = tf.exp(pis[t].pd.logp(ac) - oldpis[t].pd.logp(ac))  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
        pol_surr = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

        vf_loss = U.mean(tf.square(pis[t].vpred - ret))
        total_loss = pol_surr + vf_loss
        task_losses.append([pol_surr, vf_loss])

        var_list = pis[t].get_trainable_variables()
        lossandgrads.append(
            U.function([ob, ac, atarg, ret, lrmult], task_losses[-1] + [U.flatgrad(total_loss, var_list)]))
        task_adams.append(MpiAdam(var_list, epsilon=adam_epsilon))

        task_assign_old_eq_new.append(U.function([], [], updates=[tf.assign(oldv, newv)
                                                                  for (oldv, newv) in zipsame(oldpis[t].get_variables(),
                                                                                              pis[t].get_variables())]))
        task_compute_losses.append(U.function([ob, ac, atarg, ret, lrmult], task_losses[-1]))

        get_flats.append(U.GetFlat(var_list))
        set_from_flats.append(U.SetFromFlat(var_list))

    U.initialize()

    # match initialization of all policies
    for t in range(1, task_size):
        for i in range(len(pis[t].get_variables())):
            assign_op = pis[t].get_variables()[i].assign(pis[0].get_variables()[i].eval())
            U.get_session().run(assign_op)

    for t in range(task_size):
        task_adams[t].sync()

    # Prepare for rollouts
    # ----------------------------------------
    task_seg_gen = []
    for t in range(task_size):
        task_seg_gen.append(
            traj_segment_generator(pis[t], copy.deepcopy(env), timesteps_per_batch // task_size, stochastic=True,
                                   task_id=t))

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    max_thres_satisfied = max_threshold is None
    adjust_ratio = 0.0
    prev_avg_rew = -1000000
    revert_parameters = {}
    all_variables = []
    for t in range(task_size):
        all_variables.append(pis[t].get_variables())
        for i in range(len(all_variables[t])):
            cur_val = all_variables[t][i].eval()
            revert_parameters[all_variables[t][i].name] = cur_val
    revert_data = [0, 0, 0]

    # list for recording which parameters to share and split
    var_num = np.sum([np.prod(v.shape) for v in pis[0].get_trainable_variables()])
    share_params_list = np.zeros((task_size, var_num))

    vf_var_num = np.sum(
        [np.prod(v.shape) for v in pis[0].get_trainable_variables() if v.name.split("/")[1].startswith("vf")])
    pol_var_num = var_num - vf_var_num

    vf_share_params_list = np.zeros((task_size, vf_var_num))
    pol_share_params_list = np.zeros((task_size, pol_var_num))
    # share_params_list[1, 0:vf_var_num] = 1.0
    share_params_list[:, 0:vf_var_num] = vf_share_params_list
    share_params_list[:, vf_var_num:] = pol_share_params_list

    for t in range(task_size):
        task_assign_old_eq_new[t]()
    split_percentage_history_vf = []
    split_percentage_history_pol = []
    avg_grad_stds_buffer = []
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        segs = []
        for t in range(task_size):
            segs.append(task_seg_gen[t].__next__())

        if reward_drop_bound is not None:
            for t in range(task_size):
                lrlocal = (segs[t]["ep_lens"], segs[t]["ep_rets"])  # local values
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                lens, rews = map(flatten_lists, zip(*listoflrpairs))
                lenbuffer.extend(lens)
                rewbuffer.extend(rews)
            revert_iteration = False
            if np.mean(
                    rewbuffer) < prev_avg_rew - reward_drop_bound:  # detect significant drop in performance, revert to previous iteration
                print("Revert Iteration!!!!!")
                revert_iteration = True
            else:
                prev_avg_rew = np.mean(rewbuffer)
            logger.record_tabular("Revert Rew", prev_avg_rew)
            if revert_iteration:  # revert iteration
                for t in range(task_size):
                    for i in range(len(pis[t].get_variables())):
                        assign_op = pis[t].get_variables()[i].assign(revert_parameters[pis[t].get_variables()[i].name])
                        U.get_session().run(assign_op)
                episodes_so_far = revert_data[0]
                timesteps_so_far = revert_data[1]
                iters_so_far = revert_data[2]
                continue
            else:
                for t in range(task_size):
                    variables = pis[t].get_variables()
                    for i in range(len(variables)):
                        cur_val = variables[i].eval()
                        revert_parameters[variables[i].name] = np.copy(cur_val)
                revert_data[0] = episodes_so_far
                revert_data[1] = timesteps_so_far
                revert_data[2] = iters_so_far

        datasets = []
        for t in range(task_size):
            add_vtarg_and_adv(segs[t], gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = segs[t]["ob"], segs[t]["ac"], segs[t]["adv"], segs[t]["tdlamret"]
            vpredbefore = segs[t]["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            datasets.append(Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True))
            if hasattr(pis[t], "ob_rms"): pis[t].ob_rms.update(ob)  # update running mean/std for policy
            task_assign_old_eq_new[t]()  # set old parameter values to new parameter values

        optim_batchsize = optim_batchsize or ob.shape[0]
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        # Here we do a bunch of optimization epochs over the data
        avg_grad_stds_one = np.zeros((task_size, var_num))
        all_task_grads = np.zeros((task_size, var_num))
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            while datasets[0]._next_id <= datasets[0].n - optim_batchsize:
                avg_losses = []
                for t in range(task_size):
                    batch = datasets[t].next_batch(optim_batchsize)
                    *newlosses, g = lossandgrads[t](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                    cur_lrmult)
                    all_task_grads[t, :] += g
                    task_adams[t].update(g, optim_stepsize * cur_lrmult)
                    # synchronize the shared parameters
                    for t2 in range(task_size):
                        if t == t2 or np.sum(share_params_list[t2, :] == share_params_list[t, :]) == 0:
                            continue
                        gmask = share_params_list[t2, :] == share_params_list[t, :]
                        cur_param = task_adams[t2].getflat()
                        cur_param[gmask] = task_adams[t].getflat()[gmask]
                        task_adams[t2].setfromflat(cur_param)
                        task_adams[t2].m[gmask] = task_adams[t].m[gmask]
                        task_adams[t2].v[gmask] = task_adams[t].v[gmask]
                    avg_losses.append(newlosses)

                losses.append(np.mean(avg_losses, axis=0))

            for t in range(task_size):
                datasets[t]._next_id = 0
                datasets[t].shuffle()
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        # compute grad std
        localg = all_task_grads.astype('float64')
        globalg = np.zeros_like(localg)
        MPI.COMM_WORLD.Allreduce(localg, globalg, op=MPI.SUM)
        for t in range(task_size):
            gmask = share_params_list == t
            masked_gradient = np.ma.masked_array(globalg, mask=(1 - gmask))
            avg_grad_stds_one[t, :] += np.ma.std(masked_gradient, axis=0).data
        avg_grad_stds_buffer.append(avg_grad_stds_one)
        if len(avg_grad_stds_buffer) > 10:
            avg_grad_stds_buffer.pop(0)
        avg_grad_stds = np.mean(avg_grad_stds_buffer, axis=0)

        if iters_so_far == split_iter or (iters_so_far % split_interval == 0 and iters_so_far > 0):  # iters_so_far % 4 == 0 and iters_so_far >= 10 and iters_so_far < 50:
            logger.log("Split networks ...")
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('AVG GRAD STDS ', avg_grad_stds)

            if adapt_split:
                avg_grad_stds_vf = avg_grad_stds[:, 0:vf_var_num]
                avg_grad_stds_pol = avg_grad_stds[:, vf_var_num:]
                std_order_vf = np.zeros(np.prod(avg_grad_stds_vf.shape))
                std_order_vf[np.argsort(np.reshape(-avg_grad_stds_vf, np.prod(avg_grad_stds_vf.shape)))] = np.arange(
                    np.prod(avg_grad_stds_vf.shape))
                sorted_stds_vf = np.reshape(std_order_vf, avg_grad_stds_vf.shape)
                std_order_pol = np.zeros(np.prod(avg_grad_stds_pol.shape))
                std_order_pol[np.argsort(np.reshape(-avg_grad_stds_pol, np.prod(avg_grad_stds_pol.shape)))] = np.arange(
                    np.prod(avg_grad_stds_pol.shape))
                sorted_stds_pol = np.reshape(std_order_pol, avg_grad_stds_pol.shape)

                ########## use loss function to estimate the splitting ####################
                current_split = np.copy(share_params_list)
                current_split_vf = np.copy(vf_share_params_list)
                current_split_pol = np.copy(pol_share_params_list)
                current_params = []
                current_adam_params = []
                for p in range(task_size):
                    current_params.append(get_flats[p]())
                    current_adam_params.append([np.copy(task_adams[p].m), np.copy(task_adams[p].v)])

                estimated_performances_vf = []
                split_percents = []
                # performance of splitting with different percentages
                split_percents = [0.0, 0.05, 0.1]
                for spid in range(len(split_percents)):
                    # split value function
                    split_num = int(split_percents[spid] * int(vf_var_num))
                    for sp in range(split_num):
                        split_index = np.argwhere(sorted_stds_vf == sp)[0]
                        if avg_grad_stds_vf[sorted_stds_vf == sp] > 0:
                            # split for all tasks for now

                            split_index = np.argwhere(sorted_stds_vf == sp)[0]
                            for t in range(task_size):
                                vf_share_params_list[t, split_index[1]] = t
                    print('VF Spltting: ', np.sum(vf_share_params_list) * 1.0 / int(vf_var_num))
                    share_params_list[:, 0:vf_var_num] = vf_share_params_list
                    learning_curves = optimize_one_iter(datasets, lossandgrads, task_adams, 100, optim_stepsize,
                                                        task_size, optim_batchsize,
                                                        cur_lrmult, share_params_list)
                    # if MPI.COMM_WORLD.Get_rank() == 0:
                    #    print(spid, learning_curves)
                    # reset params
                    share_params_list = np.copy(current_split)
                    vf_share_params_list = np.copy(current_split_vf)
                    pol_share_params_list = np.copy(current_split_pol)
                    for p in range(task_size):
                        set_from_flats[p](current_params[p])
                        task_adams[p].m = np.copy(current_adam_params[p][0])
                        task_adams[p].v = np.copy(current_adam_params[p][1])
                    estimated_performances_vf.append(np.array(learning_curves)[:, 1])
                '''if MPI.COMM_WORLD.Get_rank() == 0:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    for spid in range(len(split_percents)):
                        plt.plot(np.array(estimated_performances_vf[spid]), label=str(split_percents[spid]))
                    plt.legend()
                    plt.savefig(logger.get_dir() + '/lc_' + str(iters_so_far) + '_vf.jpg')'''

                '''estimated_performances_pol = []
                split_percents = []
                # performance of splitting with different percentages
                split_percents = [0.0, 0.05, 0.1]
                for spid in range(len(split_percents)):
                    # split value function
                    split_num = int(split_percents[spid] * int(pol_var_num))
                    for sp in range(split_num):
                        split_index = np.argwhere(sorted_stds_pol == sp)[0]
                        if avg_grad_stds_pol[sorted_stds_pol == sp] > 0:
                            # split for all tasks for now
                            split_index = np.argwhere(sorted_stds_pol == sp)[0]
                            for t in range(task_size):
                                pol_share_params_list[t, split_index[1]] = t
                    print('POL Spltting: ', np.sum(pol_share_params_list) * 1.0 / int(pol_var_num))
                    share_params_list[:, vf_var_num:] = pol_share_params_list
                    learning_curves = optimize_one_iter(datasets, lossandgrads, task_adams, 100, optim_stepsize,
                                                        task_size,
                                                        optim_batchsize,
                                                        cur_lrmult, share_params_list)
                    # if MPI.COMM_WORLD.Get_rank() == 0:
                    #    print(spid, learning_curves)
                    # reset params
                    share_params_list = np.copy(current_split)
                    vf_share_params_list = np.copy(current_split_vf)
                    pol_share_params_list = np.copy(current_split_pol)
                    for p in range(task_size):
                        set_from_flats[p](current_params[p])
                        task_adams[p].m = np.copy(current_adam_params[p][0])
                        task_adams[p].v = np.copy(current_adam_params[p][1])
                    estimated_performances_pol.append(np.array(learning_curves)[:, 0])'''

                '''if MPI.COMM_WORLD.Get_rank() == 0:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    for spid in range(len(split_percents)):
                        plt.plot(np.array(estimated_performances_pol[spid]), label=str(split_percents[spid]))
                    plt.legend()
                    plt.savefig(logger.get_dir() + '/lc_' + str(iters_so_far) + '_surr.jpg')'''

            ###########################################################################

            if MPI.COMM_WORLD.Get_rank() == 0:
                if iters_so_far == split_iter:  # if
                    if rand_split:
                        avg_grad_stds[0,:] = np.random.random(avg_grad_stds[0,:].shape)
                    avg_grad_stds_pol = avg_grad_stds[:, vf_var_num:]
                    std_order_pol = np.zeros(np.prod(avg_grad_stds_pol.shape))
                    std_order_pol[
                        np.argsort(np.reshape(-avg_grad_stds_pol, np.prod(avg_grad_stds_pol.shape)))] = np.arange(
                        np.prod(avg_grad_stds_pol.shape))
                    sorted_stds_pol = np.reshape(std_order_pol, avg_grad_stds_pol.shape)
                    pol_split_num = int(split_percent * int(pol_var_num))
                    for sp in range(pol_split_num):
                        split_index = np.argwhere(sorted_stds_pol == sp)[0]
                        if avg_grad_stds_pol[sorted_stds_pol == sp] > 0:
                            # split for all tasks for now
                            split_index = np.argwhere(sorted_stds_pol == sp)[0]
                            print('splitting metric ', avg_grad_stds_pol[0, split_index[1]])
                            for t in range(task_size):
                                pol_share_params_list[t, split_index[1]] = t
                    print('POL Spltting: ', np.sum(pol_share_params_list[1,:]) * 1.0 / int(pol_var_num))
                    share_params_list[:, vf_var_num:] = pol_share_params_list
                    if split_percent == 1:
                        for t in range(task_size):
                            share_params_list[t, :] = t
                            vf_share_params_list[t, :] = t
                            pol_share_params_list[t, :] = t
                if iters_so_far % split_interval == 0 and adapt_split:
                    # check split vf and pol
                    baseline_rst_vf = np.mean(estimated_performances_vf[0][0:5]) - np.mean(
                        estimated_performances_vf[0][-5:])
                    #baseline_rst_pol = np.mean(estimated_performances_pol[0][0:5]) - np.mean(
                    #    estimated_performances_pol[0][-5:])
                    sp_perf_ratio_vf = []
                    #sp_perf_ratio_pol = []
                    for spid in range(1, len(split_percents)):
                        improv_vf = np.mean(estimated_performances_vf[0][0:5]) - np.mean(
                            estimated_performances_vf[spid][-5:])
                        #improv_pol = np.mean(estimated_performances_pol[0][0:5]) - np.mean(
                        #    estimated_performances_pol[spid][-5:])
                        sp_perf_ratio_vf.append(np.abs(improv_vf / baseline_rst_vf))
                        #sp_perf_ratio_pol.append(np.abs(improv_pol / baseline_rst_pol))
                    vf_sp_perc = 0
                    pol_sp_perc = 0
                    for id in range(len(sp_perf_ratio_vf)):
                        if sp_perf_ratio_vf[id] > 1.5:
                            vf_sp_perc = split_percents[id + 1]
                            break
                    '''for id in range(len(sp_perf_ratio_pol)):
                        if sp_perf_ratio_pol[id] > 1.5:
                            pol_sp_perc = split_percents[id + 1]
                            break'''
                    if split_percent == 1.0: # split both vf and pol in this specific case
                        vf_sp_perc = 1.0
                        adapt_split = False # disable this to save computation time
                    if vf_sp_perc != 0:
                        print(
                        '================= Split VF for ', vf_sp_perc, sp_perf_ratio_vf, ' ==========================')
                        vf_split_num = int(vf_sp_perc * int(vf_var_num))
                        for sp in range(vf_split_num):
                            split_index = np.argwhere(sorted_stds_vf == sp)[0]
                            if avg_grad_stds_vf[sorted_stds_vf == sp] > 0:
                                # split for all tasks for now
                                split_index = np.argwhere(sorted_stds_vf == sp)[0]
                                for t in range(task_size):
                                    vf_share_params_list[t, split_index[1]] = t
                        print('VF Spltting: ', np.sum(vf_share_params_list[1,:]) * 1.0 / int(vf_var_num))
                        share_params_list[:, 0:vf_var_num] = vf_share_params_list

                # plot splitted network
                param_shapes = [v.shape for v in pis[0].get_trainable_variables()]
                param_names = [v.name for v in pis[0].get_trainable_variables()]
                import matplotlib.pyplot as plt
                import matplotlib as cm
                current_id = 0
                print('plot info!!!')
                for p in range(len(param_shapes)):
                    print(param_names[p])
                    nex_id = current_id + int(np.prod(param_shapes[p]))
                    std_data = avg_grad_stds[0, current_id:nex_id]
                    split_data = share_params_list[1, current_id:nex_id]
                    current_id = nex_id
                    if len(param_shapes[p]) == 2:
                        std_data = np.reshape(std_data, param_shapes[p])
                        split_data = np.reshape(split_data, param_shapes[p])
                    else:
                        std_data = np.reshape(std_data, (param_shapes[p][0],1))
                        split_data = np.reshape(split_data, (param_shapes[p][0],1))
                    fig = plt.figure()
                    plt.imshow(std_data)
                    plt.colorbar()
                    plt.savefig(logger.get_dir() + '/'+param_names[p].replace('/','_')+'_std.jpg')
                    fig = plt.figure()
                    plt.imshow(split_data)
                    plt.colorbar()
                    plt.savefig(logger.get_dir() + '/'+param_names[p].replace('/','_')+'_split.jpg')

            MPI.COMM_WORLD.Bcast(share_params_list, root=0)
            MPI.COMM_WORLD.Bcast(vf_share_params_list, root=0)
            MPI.COMM_WORLD.Bcast(pol_share_params_list, root=0)

        split_percentage_history_vf.append(np.sum(vf_share_params_list[1,:]) * 1.0 / int(vf_var_num))
        split_percentage_history_pol.append(np.sum(pol_share_params_list[1,:]) * 1.0 / int(pol_var_num))
        print('Current Spltting (POL, VF): ', np.sum(pol_share_params_list[1,:]) * 1.0 / int(pol_var_num), np.sum(vf_share_params_list[1,:]) * 1.0 / int(vf_var_num))

        np.savetxt(logger.get_dir() + '/split_percent_history_vf.txt', split_percentage_history_vf)
        np.savetxt(logger.get_dir() + '/split_percent_history_pol.txt', split_percentage_history_pol)

        logger.log("Evaluating losses...")
        losses = []
        while datasets[0]._next_id <= datasets[0].n - optim_batchsize:
            avg_losses = []
            for t in range(task_size):
                batch = datasets[t].next_batch(optim_batchsize)
                newlosses = task_compute_losses[t](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                avg_losses.append(newlosses)
            losses.append(np.mean(avg_losses, axis=0))
        for t in range(task_size):
            datasets[t]._next_id = 0
        logger.log(fmt_row(13, np.mean(losses, axis=0)))

        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        for t in range(task_size):
            lrlocal = (segs[t]["ep_lens"], segs[t]["ep_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            if reward_drop_bound is None:
                lenbuffer.extend(lens)
                rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens) * task_size
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("Iter", iters_so_far)
        if positive_rew_enforce:
            if adjust_ratio is not None:
                logger.record_tabular("RewardAdjustRatio", adjust_ratio)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

    return pis, np.mean(rewbuffer)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
