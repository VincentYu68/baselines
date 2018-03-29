from baselines.valueiteration.utils import *
import numpy as np

def propagate_dyn_model(collected_data, dyn_rwd_model, obs_disc, act_disc):
    for trans in collected_data:
        if obs_disc(trans[0]) > obs_disc.bin_num:
            print(obs_disc(trans[0]), trans[0])
            print(obs_disc.disc_scheme)
            abc
        if obs_disc(trans[0]) not in dyn_rwd_model:
            dyn_rwd_model[obs_disc(trans[0])] = {}
        if act_disc(trans[1]) not in dyn_rwd_model[obs_disc(trans[0])]:
            dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])] = {}
        if obs_disc(trans[2]) not in dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])]:
            dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])] = [0.0, 0.0]
        dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])][0] += 1
        dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])][1] += trans[3]
    return dyn_rwd_model

def learn_model(env, obs_disc, obs_disc_dim, act_disc, act_disc_dim, state_filter, state_unfilter, policy = None, disc_policy = True, collected_data = []):
    dyn_rwd_model = {}
    collected_steps = 0

    all_state = []
    traj_num = 0
    batch_size = 500000
    additional_batch_size = 0
    total_rew = 0
    while collected_steps < batch_size:
        d = False
        env.reset()
        while not d:
            bg_step = state_filter(env.env.state_vector())
            if policy is None:
                act = env.action_space.sample()
            else:
                if disc_policy:
                    if np.random.random() < 0.9 and obs_disc(bg_step) in policy and policy[obs_disc(bg_step)] is not None:
                        act = act_disc.samp_state(policy[obs_disc(bg_step)])
                    else:
                        act = env.action_space.sample()
                else:
                    if np.random.random() < 1.0:
                        act = policy.act(True, env.env._get_obs())[0]
                    else:
                        act = env.action_space.sample()

            o,r,d,_ = env.step(act)
            ed_step = state_filter(env.env.state_vector())
            collected_data.append([bg_step, act, ed_step, r])
            collected_steps += 1
            all_state.append(bg_step)
            total_rew += r
        traj_num += 1
    print('Average rollout length: ', batch_size / traj_num)
    print('Average return: ', total_rew / traj_num)

    max_vals = np.max(all_state, axis=0)
    min_vals = np.min(all_state, axis=0)
    print(np.max(all_state, axis=0), np.min(all_state, axis=0), len(collected_data))
    for d in range(obs_disc.ndim):
        obs_disc.disc_scheme[d][1] = np.max([obs_disc.disc_scheme[d][1], max_vals[d] + 0.1])
        obs_disc.disc_scheme[d][2] = np.min([obs_disc.disc_scheme[d][2], min_vals[d] - 0.1])

    dyn_rwd_model = propagate_dyn_model(collected_data, dyn_rwd_model, obs_disc, act_disc)

    additional_data = []
    occurance_rank_stateactions = []
    for s in dyn_rwd_model.keys():
        for a in dyn_rwd_model[s].keys():
            occurence = 0
            for sn in dyn_rwd_model[s][a].keys():
                occurence += dyn_rwd_model[s][a][sn][0]
            occurance_rank_stateactions.append([occurence, [s, a]])
    occurance_rank_stateactions.sort(key=lambda ls: ls[0])

    add_steps = 0
    for i in range(len(occurance_rank_stateactions)):
        for rep in range(10):
            add_steps += 1
            s = occurance_rank_stateactions[i][1][0]
            a = occurance_rank_stateactions[i][1][1]
            samp_s = state_unfilter(obs_disc.samp_state(s))
            if np.random.random() < 0.5:
                samp_a = act_disc.samp_state(a)
            else:
                samp_a = env.action_space.sample()
            env.env.set_state_vector(samp_s)
            o,r,d,_= env.step(samp_a)
            additional_data.append([state_filter(samp_s), samp_a, state_filter(env.env.state_vector()), r])
            all_state.append(state_filter(samp_s))
        if add_steps > additional_batch_size:
            break

    dyn_rwd_model = {}
    collected_data += additional_data
    max_vals = np.max(all_state, axis=0)
    min_vals = np.min(all_state, axis=0)
    print(np.max(all_state, axis=0), np.min(all_state, axis=0), len(collected_data))
    for d in range(obs_disc.ndim):
        obs_disc.disc_scheme[d][1] = np.max([obs_disc.disc_scheme[d][1], max_vals[d] + 0.1])
        obs_disc.disc_scheme[d][2] = np.min([obs_disc.disc_scheme[d][2], min_vals[d] - 0.1])

    dyn_rwd_model = propagate_dyn_model(collected_data, dyn_rwd_model, obs_disc, act_disc)

    print('Collected data after augmentation: ', len(collected_data))

    occurances = []
    rews = []
    for s in dyn_rwd_model.keys():
        for a in dyn_rwd_model[s].keys():
            occurence = 0
            for sn in dyn_rwd_model[s][a].keys():
                occurence += dyn_rwd_model[s][a][sn][0]
                dyn_rwd_model[s][a][sn][1] /= dyn_rwd_model[s][a][sn][0]
                rews.append(dyn_rwd_model[s][a][sn][1])
            for sn in dyn_rwd_model[s][a].keys():
                dyn_rwd_model[s][a][sn][0] /= occurence
                occurances.append(occurence)
    #print(dyn_rwd_model)
    print('Occurance stat: ', np.mean(occurances), np.std(occurances), np.min(occurances), np.max(occurances))

    total_s = 1
    for d in range(obs_disc.ndim):
        total_s *= obs_disc.disc_scheme[d][0]
    print('State Occupancy Percentage: ', len(dyn_rwd_model.keys()) / total_s)
    print('Rew stat: ', np.mean(rews), np.std(rews), np.min(rews), np.max(rews))
    return dyn_rwd_model, collected_data, obs_disc


def fit_dyn_model(obs_disc, act_disc, collected_data):
    dyn_rwd_model = {}
    all_state = []
    for trans in collected_data:
        all_state.append(trans[0])
        all_state.append(trans[2])
    max_vals = np.max(all_state, axis=0)
    min_vals = np.min(all_state, axis=0)
    print(np.max(all_state, axis=0), np.min(all_state, axis=0), len(collected_data))
    for d in range(obs_disc.ndim):
        obs_disc.disc_scheme[d][1] = np.max([obs_disc.disc_scheme[d][1], max_vals[d] + 0.0001])
        obs_disc.disc_scheme[d][2] = np.min([obs_disc.disc_scheme[d][2], min_vals[d] - 0.0001])

    for trans in collected_data:
        if obs_disc(trans[0]) > obs_disc.bin_num:
            print(obs_disc(trans[0]), trans[0])
            print(obs_disc.disc_scheme)
            abc
        if obs_disc(trans[0]) not in dyn_rwd_model:
            dyn_rwd_model[obs_disc(trans[0])] = {}
        if act_disc(trans[1]) not in dyn_rwd_model[obs_disc(trans[0])]:
            dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])] = {}
        if obs_disc(trans[2]) not in dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])]:
            dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])] = [0.0, 0.0]
        dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])][0] += 1
        dyn_rwd_model[obs_disc(trans[0])][act_disc(trans[1])][obs_disc(trans[2])][1] += trans[3]

    occurances = []
    rews = []
    for s in dyn_rwd_model.keys():
        for a in dyn_rwd_model[s].keys():
            occurence = 0
            for sn in dyn_rwd_model[s][a].keys():
                occurence += dyn_rwd_model[s][a][sn][0]
                dyn_rwd_model[s][a][sn][1] /= dyn_rwd_model[s][a][sn][0]
                rews.append(dyn_rwd_model[s][a][sn][1])
            for sn in dyn_rwd_model[s][a].keys():
                dyn_rwd_model[s][a][sn][0] /= occurence
                occurances.append(occurence)
    print('Occurance stat: ', np.mean(occurances), np.std(occurances), np.min(occurances), np.max(occurances))
    print('Rew stat: ', np.mean(rews), np.std(rews), np.min(rews), np.max(rews))

    total_s = 1
    for d in range(obs_disc.ndim):
        total_s *= obs_disc.disc_scheme[d][0]
    print('State Occupancy Percentage: ', len(dyn_rwd_model.keys()) / total_s)
    return dyn_rwd_model, obs_disc



def optimize_policy(dyn_rwd_model, gamma, Vfunc = {}):
    for iter in range(1500):
        for s in dyn_rwd_model.keys():
            if s not in Vfunc:
                Vfunc[s] = 0.0
            max_nV = -1000000.0
            for a in dyn_rwd_model[s].keys():
                totalV = 0.0
                for sn in dyn_rwd_model[s][a].keys():
                    if sn not in Vfunc:
                        Vfunc[sn] = 0.0
                    totalV += dyn_rwd_model[s][a][sn][0] * (dyn_rwd_model[s][a][sn][1] + gamma * Vfunc[sn])
                if totalV > max_nV:
                    max_nV = totalV
            Vfunc[s] = max_nV
    policy = {}
    for s in dyn_rwd_model.keys():
        if s not in policy:
            policy[s] = 0.0
        max_nV = -1000000.0
        best_a = None
        for a in dyn_rwd_model[s].keys():
            totalV = 0.0
            for sn in dyn_rwd_model[s][a].keys():
                totalV += dyn_rwd_model[s][a][sn][0] * (dyn_rwd_model[s][a][sn][1] + gamma * Vfunc[sn])
            if totalV > max_nV:
                max_nV = totalV
                best_a = a
        policy[s] = best_a
    return Vfunc, policy