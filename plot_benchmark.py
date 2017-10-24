__author__ = 'yuwenhao'

import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
import json

np.random.seed(1)

if __name__ == '__main__':
    algnames = []
    for i in range(1, len(sys.argv)):
        algnames.append(sys.argv[i])

    all_data = []
    for i in range(len(algnames)):
        all_data.append([])
    for i, algname in enumerate(algnames):
        for sd in range(20):
            if os.path.exists(algname+str(sd)):
                with open(algname+str(sd)+'/progress.json') as data_file:
                    data = data_file.readlines()
                all_data[i].append([])
                for line in data:
                    pline = json.loads(line.strip())
                    all_data[i][-1].append(pline['EpRewMean'])

    colors = ['r','g','b','c','y']
    plt.figure()
    for gp in range(len(all_data)):
        for sp in range(len(all_data[gp])):
            if sp == 0:
                plt.plot(all_data[gp][sp], colors[gp], label=algnames[gp])
            else:
                plt.plot(all_data[gp][sp], colors[gp])
    plt.legend()
    plt.show()








