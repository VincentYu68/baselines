from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
import joblib

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float64')
    vpreds = np.zeros(horizon, 'float64')
    news = np.zeros(horizon, 'int32')
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
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
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

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float64')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=0.0,
        vf_iters =0,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float64, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float64, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol") or v.name.split("/")[1].startswith("logstd")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_klgrad = U.flatgrad(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float64, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    klgrads = U.function([flat_tangent, ob, ac, atarg], klgrads)
    f_tagents = U.function([flat_tangent, ob, ac, atarg], gvp)
    flat_klgrads = U.function([flat_tangent, ob, ac, atarg], flat_klgrad)

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()

    ############# test with rllab data
    '''rllab_trained_params = joblib.load('data/rllab_trained/rllab_policy_params.pkl')
    print(rllab_trained_params.keys())
    print(pi.get_variables())

    for l in range(2):
        for p in range(len(pi.get_variables())):
            if 'polfc'+str(l+1) in pi.get_variables()[p].name:
                if 'w:' in pi.get_variables()[p].name:
                    assign_op = pi.get_variables()[p].assign(rllab_trained_params['hidden_'+str(l)+'.W'])
                    U.get_session().run(assign_op)
                if 'b:' in pi.get_variables()[p].name:
                    assign_op = pi.get_variables()[p].assign(rllab_trained_params['hidden_'+str(l)+'.b'])
                    U.get_session().run(assign_op)
    for p in range(len(pi.get_variables())):
        if 'polfinal' in pi.get_variables()[p].name:
            if 'w:' in pi.get_variables()[p].name:
                assign_op = pi.get_variables()[p].assign(rllab_trained_params['output.W'])
                U.get_session().run(assign_op)
            if 'b:' in pi.get_variables()[p].name:
                assign_op = pi.get_variables()[p].assign(rllab_trained_params['output.b'])
                U.get_session().run(assign_op)
        if 'logstd' in pi.get_variables()[p].name:
            assign_op = pi.get_variables()[p].assign(np.reshape(rllab_trained_params['output_log_std.param'], (1, len(rllab_trained_params['output_log_std.param']))))
            U.get_session().run(assign_op)


    ob = np.arange(21)
    print(pi.act(False, ob))

    input_data = joblib.load('data/rllab_trained/input_data.pkl')
    #print(input_data)
    assign_old_eq_new()

    flat_pm = get_flat()
    set_from_flat(flat_pm + 0.01)

    with timed("computegrad"):
        *lossbefore, g = compute_lossandgrad(input_data[0], input_data[1], input_data[2])
    #print(g)
    #print(pi.get_trainable_variables())

    fvpargs = [arr for arr in input_data]
    def fisher_vector_product(p):
        return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

    #print(klgrads(g, *fvpargs))
    #print(lossbefore[1])'''

    # manually compute kl divergence for comparison
    '''all_obs = input_data[0]
    old_mean_vals = []
    new_mean_vals = []
    for ob in all_obs:
        old_mean_vals.append(oldpi.act(False, ob)[0])
        new_mean_vals.append(pi.act(False, ob)[0])
    klval = 0

    for o in range(len(new_mean_vals)):
        old_std = np.exp(np.array([0.0,0.0,0.0,0.0,0.0]))
        new_std = np.exp(np.array([0.01,0.01,0.01,0.01,0.01]))

        numerator = np.square(old_mean_vals[o] - new_mean_vals[o]) + np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std)
        klval += np.sum(numerator / denominator + np.array([0.01,0.01,0.01,0.01,0.01]) - np.array([0.0,0.0,0.0,0.0,0.0]))
    klval /= len(new_mean_vals)
    print('brute force kl', klval)
    print('kl error: ', np.abs(klval - lossbefore[1]))
    abc'''

    #print('F GVP :', np.dot(flat_klgrads(g, *fvpargs), g))
    #print('kl grads ', flat_klgrads(g, *fvpargs))

    # use finite difference to estimate the kl grads:
    '''current_params = np.copy(get_flat())
    kl_grad_fd = np.zeros(len(current_params))
    for fdi in range(len(kl_grad_fd)):
        perturbed_params = np.copy(current_params)
        perturbed_params[fdi] += 1e-10
        set_from_flat(perturbed_params)
        *lossbefore, g = compute_lossandgrad(input_data[0], input_data[1], input_data[2])
        rval = lossbefore[1]
        perturbed_params[fdi] -= 2e-10
        set_from_flat(perturbed_params)
        *lossbefore, g = compute_lossandgrad(input_data[0], input_data[1], input_data[2])
        lval = lossbefore[1]

        kl_grad_fd[fdi] = (rval - lval) / 2e-10
    print('KL div finite diff ', kl_grad_fd)'''
    ###################################################

    #print('kl div grad error: ', np.linalg.norm(kl_grad_fd - flat_klgrads(g, *fvpargs)))

    #print(g)
    #print(fisher_vector_product(g))
    '''with timed("cg"):
        stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
    print(stepdir)
    print(stepdir.dtype)

    abc'''
    ############# test with rllab data

    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    while True:
        print(pi.get_variables()[-1].name, pi.get_variables()[-1].eval())
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]