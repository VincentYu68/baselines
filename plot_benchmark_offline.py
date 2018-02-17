__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json

np.random.seed(1)

if __name__ == '__main__':
    # setup for hopper forward backward 1500 samples
    plot_setup = [
        [['ppo_DartHopper-v11_split_0_1.0_2tasktorsomass095_2000'], 'SplitAll', 'y'],
        [['ppo_DartHopper-v11_split_1000_0.0_2tasktorsomass095_2000'], 'share all', 'b'],

        [['ppo_DartHopper-v11_split_10_0.2_2tasktorsomass095_2000'], 'Split_10_0.2'],
        [['ppo_DartHopper-v11_split_10_0.4_2tasktorsomass095_2000'], 'Split_10_0.4'],
        [['ppo_DartHopper-v11_split_10_0.6_2tasktorsomass095_2000'], 'Split_10_0.6'],

        #[['ppo_DartHopper-v11_split_40_0.2_2tasktorsomass05_2000'], 'Split_40_0.2'],
        [['ppo_DartHopper-v11_split_40_0.4_2tasktorsomass095_2000'], 'Split_40_0.4'],
        [['ppo_DartHopper-v11_split_40_0.6_2tasktorsomass095_2000'], 'Split_40_0.6'],

        [['ppo_DartHopper-v11_split_90_0.2_2tasktorsomass095_2000'], 'Split_90_0.2'],
        [['ppo_DartHopper-v11_split_90_0.4_2tasktorsomass095_2000'], 'Split_90_0.4'],
        [['ppo_DartHopper-v11_split_90_0.6_2tasktorsomass095_2000'], 'Split_90_0.6'],
    ]

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_0_1.0_2task_forwardbackward_2000'], 'SplitAll'],
        [['ppo_DartHopper-v11_split_30_0.2_2task_forwardbackward_2000'], 'Split_30_0.2'],
        [['ppo_DartHopper-v11_split_30_0.4_2task_forwardbackward_2000'], 'Split_30_0.4'],
        [['ppo_DartHopper-v11_split_30_0.6_2task_forwardbackward_2000'], 'Split_30_0.6'],
        [['ppo_DartHopper-v11_split_30_0.8_2task_forwardbackward_2000'], 'Split_30_0.8'],

        [['ppo_DartHopper-v11_split_80_0.2_2task_forwardbackward_2000'], 'Split_80_0.2'],
        [['ppo_DartHopper-v11_split_80_0.4_2task_forwardbackward_2000'], 'Split_80_0.4'],
        [['ppo_DartHopper-v11_split_80_0.6_2task_forwardbackward_2000'], 'Split_80_0.6'],
        [['ppo_DartHopper-v11_split_80_0.8_2task_forwardbackward_2000'], 'Split_80_0.8'],

        [['ppo_DartHopper-v11_split_130_0.2_2task_forwardbackward_2000'], 'Split_130_0.2'],
        [['ppo_DartHopper-v11_split_130_0.4_2task_forwardbackward_2000'], 'Split_130_0.4'],
        [['ppo_DartHopper-v11_split_130_0.6_2task_forwardbackward_2000'], 'Split_130_0.6'],
        [['ppo_DartHopper-v11_split_130_0.8_2task_forwardbackward_2000'], 'Split_130_0.8'],

        [['ppo_DartHopper-v11_split_10000_0.0_2task_forwardbackward_2000'], 'share all'],
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_0_1.0_3models_3000'], 'SplitAll', 'y'],
        [['ppo_DartHopper-v11_split_1000_0.0_3models_3000'], 'share all', 'b'],

        [['ppo_DartHopper-v11_split_20_0.2_3models_3000'], 'Split_20_0.2'],
        [['ppo_DartHopper-v11_split_20_0.4_3models_3000'], 'Split_20_0.4'],
        [['ppo_DartHopper-v11_split_20_0.6_3models_3000'], 'Split_20_0.6'],

        [['ppo_DartHopper-v11_split_50_0.2_3models_3000'], 'Split_50_0.2'],
        [['ppo_DartHopper-v11_split_50_0.4_3models_3000'], 'Split_50_0.4'],
        [['ppo_DartHopper-v11_split_50_0.6_3models_3000'], 'Split_50_0.6'],

        [['ppo_DartHopper-v11_split_100_0.2_3models_3000'], 'Split_100_0.2'],
        [['ppo_DartHopper-v11_split_100_0.4_3models_3000'], 'Split_100_0.4'],
        [['ppo_DartHopper-v11_split_100_0.6_3models_3000'], 'Split_100_0.6'],
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_1000_0.0_2modelfric_2000'], 'SplitAll', 'y'],
        [['ppo_DartHopper-v11_split_0_1.0_2modelfric_2000'], 'share all', 'b'],

        [['ppo_DartHopper-v11_split_10_0.2_2modelfric_2000'], 'Split_10_0.2'],
        [['ppo_DartHopper-v11_split_10_0.4_2modelfric_2000'], 'Split_10_0.4'],
        [['ppo_DartHopper-v11_split_10_0.6_2modelfric_2000'], 'Split_10_0.6'],
        [['ppo_DartHopper-v11_split_10_0.8_2modelfric_2000'], 'Split_10_0.8'],

        [['ppo_DartHopper-v11_split_40_0.2_2modelfric_2000'], 'Split_40_0.2'],
        [['ppo_DartHopper-v11_split_40_0.4_2modelfric_2000'], 'Split_40_0.4'],
        [['ppo_DartHopper-v11_split_40_0.6_2modelfric_2000'], 'Split_40_0.6'],
        [['ppo_DartHopper-v11_split_40_0.8_2modelfric_2000'], 'Split_40_0.8'],

        [['ppo_DartHopper-v11_split_90_0.2_2modelfric_2000'], 'Split_90_0.2'],
        [['ppo_DartHopper-v11_split_90_0.4_2modelfric_2000'], 'Split_90_0.4'],
        [['ppo_DartHopper-v11_split_90_0.6_2modelfric_2000'], 'Split_90_0.6'],
        [['ppo_DartHopper-v11_split_90_0.8_2modelfric_2000'], 'Split_90_0.8'],
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_0_1.0_2models_2000'], 'SplitAll', 'y'],
        [['ppo_DartHopper-v11_split_1000_0.0_2models_2000'], 'share all', 'b'],

        [['ppo_DartHopper-v11_split_10_0.2_2models_2000'], 'Split_10_0.2'],
        [['ppo_DartHopper-v11_split_10_0.4_2models_2000'], 'Split_10_0.4'],
        [['ppo_DartHopper-v11_split_10_0.6_2models_2000'], 'Split_10_0.6'],

        [['ppo_DartHopper-v11_split_40_0.2_2models_2000'], 'Split_40_0.2'],
        [['ppo_DartHopper-v11_split_40_0.4_2models_2000'], 'Split_40_0.4'],
        [['ppo_DartHopper-v11_split_40_0.6_2models_2000'], 'Split_40_0.6'],

        [['ppo_DartHopper-v11_split_90_0.2_2models_2000'], 'Split_90_0.2'],
        [['ppo_DartHopper-v11_split_90_0.4_2models_2000'], 'Split_90_0.4'],
        [['ppo_DartHopper-v11_split_90_0.6_2models_2000'], 'Split_90_0.6'],
    ]'''

    all_data = []
    step_sofar = []
    legend_names = []
    colors = []
    for i, one_trial in enumerate(plot_setup):
        legend_names.append(one_trial[1])
        trial_names = one_trial[0]
        if len(one_trial) == 3:
            colors.append(one_trial[2])
        else:
            colors.append('')
        avg_data = []
        avg_steps = []
        for n in range(len(trial_names)):
            name = trial_names[n]
            # augment with other random seeds
            for i in range(2, 20):
                augname = name.replace('-v11', '-v1'+str(i))
                trial_names.append(augname)
        for i, name in enumerate(trial_names):
            one_data = []
            one_data_step = []
            filepath = 'data/split_net/' + name
            if os.path.exists(filepath):
                with open(filepath+'/progress.json') as data_file:
                    data = data_file.readlines()
                for line in data:
                    pline = json.loads(line.strip())
                    one_data.append(pline['EpRewMean'])
                    one_data_step.append(pline['TimestepsSoFar'])
            if len(one_data) > 0:
                avg_data.append(one_data)
                avg_steps.append(one_data_step)

        print(np.array(avg_data).shape)
        min_rolloutlength = np.min([len(arr) for arr in avg_data])
        for i in range(len(avg_data)):
            avg_data[i] = avg_data[i][0:min_rolloutlength]
            avg_steps[i] = avg_steps[i][0:min_rolloutlength]
        all_data.append(np.mean(avg_data, axis=0))
        step_sofar.append(np.mean(avg_steps, axis=0))

    #colors = ['g','r','b','c','y']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for sp in range(len(all_data)):
        #if len(all_data[sp]) > 1000:
        #    all_data[sp] = all_data[sp][0:1000]
        if len(colors[sp]) == 0:
            c = (1.0 - sp / len(all_data), sp / len(all_data), 0.0)
        else:
            c = colors[sp]
        ax.plot(step_sofar[sp], all_data[sp], linewidth=2, color=c, label=legend_names[sp], alpha=0.5*sp/len(all_data)+0.5)
    plt.legend()

    plt.title('Hopper 2 models cap ellip', fontsize=14)

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.show()









