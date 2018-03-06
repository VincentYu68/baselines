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
    '''plot_setup = [
        [['ppo_DartHopper-v11_split_0_1.0_10000_0_2taskfwdbwd_2000'], 'SplitAll', 'y'],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_2taskfwdbwd_2000'], 'share all', 'b'],

        [['ppo_DartHopper-v11_split_10_0.05_5_0_2taskfwdbwd_2000'], 'Continuous split'],
        [['ppo_DartHopper-v11_split_10_0.0_5_1_2taskfwdbwd_2000'], 'Adapt Split'],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_fwdbwdapp_2000'], 'Append']
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_torso_0951_1000'], 'sp=0%', 'b'],
            [['ppo_DartHopper-v11_split_1_1.0_10000_0_torso_0951_1000'], 'jt=0, sp=100%', 'y'],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_0951_new_random_1000'], 'jt=400k, sp=25% random', (0.1, 0.0, 0.8)],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_torso_0951_new_app_1000'], 'append', 'append', (0.1, 0.6, 0.6)],
            [['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_0951_new_1000'], 'jt=400k, sp=25%', (0.0,0.6,0.0)],
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_torso_051_1000'], 'sp=0%', 'b'],
        [['ppo_DartHopper-v11_split_1_1.0_10000_0_torso_051_1000'], 'jt=0, sp=100%', 'y'],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_051_new_random_1000'], 'jt=400k, sp=25% random', (0.1, 0.0, 0.8)],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_torso_051_new_app_1000'], 'append', 'append', (0.1, 0.6, 0.6)],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_051_new_1000'], 'jt=400k, sp=25%', (0.0,0.6,0.0)],
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_torso_011_1000'], 'sp=0%', 'b'],
        [['ppo_DartHopper-v11_split_1_1.0_10000_0_torso_011_1000'], 'jt=0, sp=100%', 'y'],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_011_new3_random_1000'], 'jt=400k, sp=25% random', (0.1, 0.0, 0.8)],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_torso_011_new_app_1000'], 'append', 'append', (0.1, 0.6, 0.6)],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_011_new_1000'], 'jt=400k, sp=25%', (0.0,0.6,0.0)],
        #[['ppo_DartHopper-v11_split_100_0.25_10000_0_torso_011_new_ref_1000'], '100 0.25 rev'],
    ]'''


    '''plot_setup = [
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_3models_new3_1000'], 'sp=0%', 'b'],
        [['ppo_DartHopper-v11_split_1_1.0_10000_0_3models_new2_1000'], 'jt=0, sp=100%', 'y'],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_3models_new3_random_1000'], 'jt=400k, sp=25% random',
         (0.1, 0.0, 0.8)],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_3models_new3_app_1000'], 'append', (0.1, 0.6, 0.6)],
        [['ppo_DartHopper-v11_split_20_0.05_10000_0_3models_new4_1000'], 'jt=80k, sp=5%', (0.3,0.0,0.3)],
        [['ppo_DartHopper-v11_split_20_0.25_10000_0_3models_new4_1000'], 'jt=80k, sp=25%', (0.6,0.0,0.6)],
        [['ppo_DartHopper-v11_split_20_0.5_10000_0_3models_new4_1000'], 'jt=80k, sp=50%', (0.9,0.0,0.9)],
        [['ppo_DartHopper-v11_split_100_0.05_10000_0_3models_new4_1000'], 'jt=400k, sp=5%', (0.0,0.3,0.0)],
        [['ppo_DartHopper-v11_split_100_0.25_10000_0_3models_new4_1000'], 'jt=400k, sp=25%', (0.0,0.6,0.0)],
        [['ppo_DartHopper-v11_split_100_0.5_10000_0_3models_new3_1000'], 'jt=400k, sp=50%', (0.0,0.9,0.0)],
        [['ppo_DartHopper-v11_split_200_0.05_10000_0_3models_new4_1000'], 'jt=800k, sp=5%', (0.4,0.0,0.0)],
        [['ppo_DartHopper-v11_split_200_0.25_10000_0_3models_new4_1000'], 'jt=800k, sp=25%', (0.6,0.0,0.0)],
        [['ppo_DartHopper-v11_split_200_0.5_10000_0_3models_new4_1000'], 'jt=800k, sp=50%', (0.9,0.0,0.0)],
        #[['ppo_DartHopper-v11_split_100_0.05_10000_0_3models_new4_random_1000'], 'jt 400ksp 5p random', (0,1,1)]
    ]'''

    '''plot_setup = [
        [['ppo_DartHopper-v11_split_1_1.0_10000_0_3models_new2_2000'], 'SplitAll', 'y'],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_3models_new2_2000'], 'share all', 'b'],
        [['ppo_DartHopper-v11_split_10_0.05_10000_0_3models_new2_2000'], '10 0.05', (0.3,0.0,0.3)],
        [['ppo_DartHopper-v11_split_10_0.25_10000_0_3models_new2_2000'], '10 0.25', (0.6,0.0,0.6)],
        [['ppo_DartHopper-v11_split_50_0.05_10000_0_3models_new2_2000'], '50 0.05', (0.0,0.3,0.0)],
        [['ppo_DartHopper-v11_split_50_0.25_10000_0_3models_new2_2000'], '50 0.25', (0.0,0.6,0.0)],
        [['ppo_DartHopper-v11_split_50_0.25_10000_0_3models_random_2000'], '50 0.25 random', (0.6,0.0,0.0)],

        #[['ppo_DartHopper-v11_split_10_0.05_10_1_3models_new2_2000'], '10 0.05'],
        #[['ppo_DartHopper-v11_split_10_0.25_10_1_3models_new2_2000'], '10 0.25'],
        #[['ppo_DartHopper-v11_split_10_0.6_5_1_3models_new_2000'], '10 0.6'],
        #[['ppo_DartHopper-v11_split_50_0.05_10_1_3models_new2_2000'], '50 0.05'],
        #[['ppo_DartHopper-v11_split_50_0.25_10_1_3models_new2_2000'], '50 0.25'],
        #[['ppo_DartHopper-v11_split_50_0.6_5_1_3models_new_2000'], '50 0.6'],
        #[['ppo_DartHopper-v11_split_100_0.05_5_1_3models_new_2000'], '100 0.05'],
        #[['ppo_DartHopper-v11_split_100_0.25_5_1_3models_new_2000'], '100 0.25'],
        #[['ppo_DartHopper-v11_split_100_0.6_5_1_3models_new_2000'], '100 0.6'],
        [['ppo_DartHopper-v11_split_1000_0.0_10000_0_3models_new_app_2000'], 'app', (0.9,0.0,0.0)]
    ]'''

    '''plot_setup = [
        [['ppo_DartReacher3d-v11_split_1000_0.0_10000_0_3modelsr35_diffaxis_new_4000'], 'share all', 'b'],
        #[['ppo_DartReacher3d-v11_split_1_1.0_10000_0_3modelsr35_diffaxis_new_4000'], 'split all', 'y'],
        [['ppo_DartReacher3d-v11_split_5_0.05_10000_0_3modelsr35_diffaxis_new_4000'], '10 0.05', (0.3,0.0,0.3)],
        #[['ppo_DartReacher3d-v11_split_5_0.25_10000_0_3modelsr35_diffaxis_new_4000'], '10 0.25', (0.6,0.0,0.6)],
        #[['ppo_DartReacher3d-v11_split_5_0.5_10000_0_3modelsr35_diffaxis_new_4000'], '10 0.5', (0.9,0.0,0.9)],
        [['ppo_DartReacher3d-v11_split_25_0.05_10000_0_3modelsr35_diffaxis_new_4000'], '50 0.05', (0.0,0.3,0.0)],
        [['ppo_DartReacher3d-v11_split_25_0.25_10000_0_3modelsr35_diffaxis_new_4000'], '50 0.25', (0.0,0.6,0.0)],
        [['ppo_DartReacher3d-v11_split_25_0.05_10000_0_3modelsr35_diffaxis_new_4000'], '50 0.5', (0.0,0.9,0.0)],
        [['ppo_DartReacher3d-v11_split_50_0.05_10000_0_3modelsr35_diffaxis_new_4000'], '100 0.05', (0.4,0.0,0.0)],
        [['ppo_DartReacher3d-v11_split_50_0.25_10000_0_3modelsr35_diffaxis_new_4000'], '100 0.25', (0.6,0.0,0.0)],
        #[['ppo_DartReacher3d-v11_split_50_0.5_10000_0_3modelsr35_diffaxis_new_4000'], '100 0.5', (0.9, 0.0,0.0)],
    ]'''

    '''plot_setup = [
        [['ppo_DartReacher3d-v11_split_1000_0.0_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'share all', 'b'],
        [['ppo_DartReacher3d-v11_split_1000_0.0_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_append_1000'], 'append', 'r'],
        [['ppo_DartReacher3d-v11_split_20_0.05_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=80k, sp=5%', (0.3,0.0,0.3)],
        [['ppo_DartReacher3d-v11_split_20_0.25_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=80k, sp=25%', (0.6,0.0,0.6)],
        [['ppo_DartReacher3d-v11_split_20_0.5_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=80k, sp=50%', (0.9,0.0,0.9)],
        [['ppo_DartReacher3d-v11_split_100_0.05_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=400k, sp=5%', (0.0,0.3,0.0)],
        [['ppo_DartReacher3d-v11_split_100_0.25_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=400k, sp=25%', (0.0,0.6,0.0)],
        [['ppo_DartReacher3d-v11_split_100_0.5_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=400k, sp=50%', (0.0,0.9,0.0)],
        [['ppo_DartReacher3d-v11_split_200_0.05_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=800k, sp=5%', (0.4,0.0,0.0)],
        [['ppo_DartReacher3d-v11_split_200_0.25_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=800k, sp=25%', (0.6,0.0,0.0)],
        [['ppo_DartReacher3d-v11_split_200_0.5_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'jt=800k, sp=50%', (0.9,0.0,0.0)],
        [['ppo_DartReacher3d-v11_split_1_1.0_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'split all', 'y'],
        #[['ppo_DartReacher3d-v11_split_1_1.0_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_1000'], 'split all', 'y'],
        [['ppo_DartReacher3d-v11_split_100_0.25_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_random_1000'], '400k 0.25 random', (0.1, 0.0, 0.8)],
        [['ppo_DartReacher3d-v11_split_200_0.25_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_random_1000'], '800k 0.25 random', (0.3, 0.5, 0.2)],
        [['ppo_DartReacher3d-v11_split_20_0.25_10000_0_3modelsr05_diffaxis_new_jointlimit_revcont_random_1000'],
         '80k 0.25 random', (0.9, 0.5, 0.2)],
    ]'''

    plot_setup = [
        [['ppo_DartReacher3d-v11_split_1000_0.0_10000_0_1model_tracking_rev_1000'], 'share all', 'b'],
        [['ppo_DartReacher3d-v11_split_1000_0.0_10000_0_1model_tracking_rev_append_1000'], 'append', (0.1, 0.6, 0.6) ],
        [['ppo_DartReacher3d-v11_split_1_1.0_10000_0_2model_tracking_1000'], 'split all', 'y'],
        [['ppo_DartReacher3d-v11_split_100_0.25_10000_0_1model_tracking_rev_random_1000'], '400k 0.25 random', (0.1, 0.0, 0.8)],
        #[['ppo_DartReacher3d-v11_split_20_0.05_10000_0_1model_tracking_rev_1000'], '20 0.05', (0.3,0.0,0.3)],
        #[['ppo_DartReacher3d-v11_split_20_0.25_10000_0_1model_tracking_rev_1000'], '20 0.25', (0.6,0.0,0.6)],
        #[['ppo_DartReacher3d-v11_split_20_0.5_10000_0_1model_tracking_rev_1000'], '20 0.5', (0.9,0.0,0.9)],
        [['ppo_DartReacher3d-v11_split_100_0.05_10000_0_1model_tracking_rev_1000'], '100 0.05'],
        [['ppo_DartReacher3d-v11_split_100_0.25_10000_0_1model_tracking_rev_1000'],'100 0.25', (0.0,0.6,0.0)],
        [['ppo_DartReacher3d-v11_split_100_0.5_10000_0_1model_tracking_rev_1000'], '100 0.5'],
        #[['ppo_DartReacher3d-v11_split_200_0.05_10000_0_1model_tracking_rev_1000'], '200 0.05'],
        #[['ppo_DartReacher3d-v11_split_200_0.25_10000_0_1model_tracking_rev_1000'], '200 0.25'],
        #[['ppo_DartReacher3d-v11_split_200_0.5_10000_0_1model_tracking_rev_1000'], '200 0.5']
    ]

    '''plot_setup = [
        #[['ppo_DartWalker2d-v11_split_1_1.0_10000_0_fwdbwd_2000'], 'split all no vf', 'y'],
        [['ppo_DartWalker2d-v11_split_1000_0.0_10000_0_fwdbwd_2000'], 'sp=0%', 'b'],
        [['ppo_DartWalker2d-v11_split_1_1.0_10000_0_fwdbwd_vfsplit_2000'], 'jt=0, sp=100%', 'y'],
        [['ppo_DartWalker2d-v11_split_50_0.25_10000_0_fwdbwd_random_2000'], 'jt=400k, sp=25% Random', (0.1, 0.0, 0.8)],
        [['ppo_DartWalker2d-v11_split_1000_0.0_10000_0_fwdbwd_app_2000'], 'append', (0.1, 0.6, 0.6)],

        [['ppo_DartWalker2d-v11_split_10_0.05_10000_0_fwdbwd_2000'], 'jt=80k, sp=5%', (0.3,0.0,0.3)],
        [['ppo_DartWalker2d-v11_split_10_0.25_10000_0_fwdbwd_2000'], 'jt=80k, sp=25%', (0.6,0.0,0.6)],
        [['ppo_DartWalker2d-v11_split_10_0.5_10000_0_fwdbwd_2000'], 'jt=80k, sp=50%', (0.9,0.0,0.9)],
        [['ppo_DartWalker2d-v11_split_50_0.05_10000_0_fwdbwd_2000'], 'jt=400k, sp=5%', (0.0,0.3,0.0)],
        [['ppo_DartWalker2d-v11_split_50_0.25_10000_0_fwdbwd_2000'], 'jt=400k, sp=25%', (0.0,0.6,0.0)],
        [['ppo_DartWalker2d-v11_split_50_0.5_10000_0_fwdbwd_2000'], 'jt=400k, sp=50%', (0.0,0.9,0.0)],
        [['ppo_DartWalker2d-v11_split_100_0.05_10000_0_fwdbwd_2000'], 'jt=800k, sp=5%', (0.4,0.0,0.0)],
        [['ppo_DartWalker2d-v11_split_100_0.25_10000_0_fwdbwd_2000'], 'jt=800k, sp=25%', (0.6,0.0,0.0)],
        [['ppo_DartWalker2d-v11_split_100_0.5_10000_0_fwdbwd_2000'], 'jt=800k, sp=50%', (0.9,0.0,0.0)],
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
                    if pline['TimestepsSoFar'] > 3000000:
                        break
                print(name, len(one_data))
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
        ax.plot(step_sofar[sp], all_data[sp], linewidth=2, color=c, label=legend_names[sp], alpha=0.8)
        print(legend_names[sp], all_data[sp][-1])
    plt.legend()

    plt.title('Reacher', fontsize=14)

    plt.xlabel("Sample Num", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.show()









