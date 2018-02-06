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
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    env.env.env.enforce_task_id = task_id
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
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
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "task_ids":task_ids}
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
                t -= (cur_ep_len+1)
            cur_ep_ret = 0
            cur_ep_len = 0
            env.env.env.enforce_task_id = task_id
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        return_threshold = None, # termiante learning if reaches return_threshold
        op_after_init = None,
        init_policy_params = None,
        policy_scope=None,
        max_threshold=None,
        positive_rew_enforce = False,
        reward_drop_bound = None,
        min_iters = 0,
        ref_policy_params = None,
         rollout_length_thershold = None
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

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pis[0].pdtype.sample_placeholder([None])

    loss_names = ["pol_surr", "vf_loss"]
    task_losses = []
    task_assign_old_eq_new = []
    task_compute_losses = []
    task_adams = []
    lossandgrads = []
    for t in range(task_size):
        ratio = tf.exp(pis[t].pd.logp(ac) - oldpis[t].pd.logp(ac)) # pnew / pold
        surr1 = ratio * atarg # surrogate from conservative policy iteration
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
        pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

        vf_loss = U.mean(tf.square(pis[t].vpred - ret))
        total_loss = pol_surr + vf_loss
        task_losses.append([pol_surr, vf_loss])

        var_list = pis[t].get_trainable_variables()
        lossandgrads.append(U.function([ob, ac, atarg, ret, lrmult], task_losses[-1] + [U.flatgrad(total_loss, var_list)]))
        task_adams.append(MpiAdam(var_list, epsilon=adam_epsilon))

        task_assign_old_eq_new.append(U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpis[t].get_variables(), pis[t].get_variables())]))
        task_compute_losses.append(U.function([ob, ac, atarg, ret, lrmult], task_losses[-1]))

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
        task_seg_gen.append(traj_segment_generator(pis[t], copy.deepcopy(env), timesteps_per_batch//task_size, stochastic=True, task_id=t))

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

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
    var_num = np.sum([np.prod(v.shape) for v in pis[t].get_trainable_variables()])
    share_params_list = np.zeros((task_size, var_num))
    #share_params_list[1, :] = 1.0

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
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

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
                    rewbuffer) < prev_avg_rew - 50:  # detect significant drop in performance, revert to previous iteration
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
            vpredbefore = segs[t]["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            datasets.append(Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True))
            if hasattr(pis[t], "ob_rms"): pis[t].ob_rms.update(ob) # update running mean/std for policy
            task_assign_old_eq_new[t]() # set old parameter values to new parameter values

        optim_batchsize = optim_batchsize or ob.shape[0]
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        # Here we do a bunch of optimization epochs over the data
        avg_grad_stds = np.zeros((task_size, var_num))
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            while datasets[0]._next_id <= datasets[0].n - optim_batchsize:
                avg_losses = []
                task_gradients = []
                for t in range(task_size):
                    batch = datasets[t].next_batch(optim_batchsize)
                    *newlosses, g = lossandgrads[t](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    task_gradients.append(g)
                    avg_losses.append(newlosses)
                task_gradients = np.array(task_gradients)

                combined_gradients = np.zeros(task_gradients.shape)
                for t in range(task_size):
                    gmask = share_params_list == t
                    gt = np.tile(np.sum(task_gradients * gmask, axis=0) / np.clip(np.sum(gmask, axis=0), 0.001, task_size+1), (task_size, 1))
                    combined_gradients += gt * gmask

                # compute grad std
                localg = task_gradients.astype('float32')
                globalg = np.zeros_like(localg)
                MPI.COMM_WORLD.Allreduce(localg, globalg, op=MPI.SUM)

                for t in range(task_size):
                    gmask = share_params_list == t
                    masked_gradient = np.ma.masked_array(globalg, mask=(1-gmask))
                    avg_grad_stds[t, :] += np.ma.std(masked_gradient, axis=0).data
                #print(combined_gradients[:, 10:15])
                #print('============')
                for t in range(task_size):
                    task_adams[t].update(combined_gradients[t, :], optim_stepsize * cur_lrmult)
                losses.append(np.mean(avg_losses, axis=0))

            for t in range(task_size):
                datasets[t]._next_id = 0
            logger.log(fmt_row(13, np.mean(losses, axis=0)))
        if iters_so_far == 50:#iters_so_far % 4 == 0 and iters_so_far >= 10 and iters_so_far < 50:
            logger.log("Split networks ...")
            std_order = np.zeros(np.prod(avg_grad_stds.shape))
            std_order[np.argsort(np.reshape(-avg_grad_stds, np.prod(avg_grad_stds.shape)))] = np.arange(np.prod(avg_grad_stds.shape))
            sorted_stds = np.reshape(std_order, avg_grad_stds.shape)
            # split top 50%
            split_num = int(0.5 * int(var_num))
            splitted = 0
            for sp in range(split_num):
                split_index = np.argwhere(sorted_stds == sp)[0]
                if share_params_list[split_index[0], split_index[1]] == 1:
                    print('wrong')
                    abc

                if avg_grad_stds[sorted_stds==sp] > 0:
                    splitted += 1
                    # split for all tasks for now
                    split_index = np.argwhere(sorted_stds==sp)[0]
                    for t in range(task_size):
                        share_params_list[t, split_index[1]] = t
        print('Current Spltting: ', np.sum(share_params_list)*1.0 / int(var_num))


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

        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
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
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        if max_threshold is not None:
            print('Current max return: ', np.max(rewbuffer))
            if np.max(rewbuffer) > max_threshold:
                max_thres_satisfied = True
            else:
                max_thres_satisfied = False

        return_threshold_satisfied = True
        if return_threshold is not None:
            if not(np.mean(rewbuffer) > return_threshold and iters_so_far > min_iters):
                return_threshold_satisfied = False
    return pis, np.mean(rewbuffer)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
