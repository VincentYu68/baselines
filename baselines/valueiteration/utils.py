from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

class bin_disc:
    def __init__(self, disc_scheme):
        self.disc_scheme = disc_scheme # scheme for each dimension is organized as : [size, max, min]
        self.ndim = len(self.disc_scheme)

    def __call__(self, x):
        assert(len(x) == self.ndim) # dimension must match
        idx = 0
        base_mult = 1
        for d in range(len(x)):
            vmax = self.disc_scheme[d][1]
            vmin = self.disc_scheme[d][2]
            size = self.disc_scheme[d][0]
            idd = np.floor((np.clip(x[d], vmin, vmax) - vmin) / (vmax-vmin) * size)
            idx += idd * base_mult
            base_mult *= size
        return int(idx)

    def get_midstate(self, idx): # get the state corresponding to of the id
        restidx = idx
        state = [0.0] * self.ndim
        for i in range(self.ndim-1, -1, -1):
            mult = 1.0
            for d in range(i):
                mult *= self.disc_scheme[d][0]
            idi = np.floor(restidx / mult)
            restidx = restidx - idi * mult
            vmax = self.disc_scheme[i][1]
            vmin = self.disc_scheme[i][2]
            size = self.disc_scheme[i][0]
            state[i] = (vmax-vmin) * (idi+0.5)/size+vmin
        return np.array(state)

    def samp_state(self, idx):
        restidx = idx
        state = [0.0] * self.ndim
        for i in range(self.ndim - 1, -1, -1):
            mult = 1.0
            for d in range(i):
                mult *= self.disc_scheme[d][0]
            idi = np.floor(restidx / mult)
            restidx = restidx - idi * mult
            vmax = self.disc_scheme[i][1]
            vmin = self.disc_scheme[i][2]
            size = self.disc_scheme[i][0]
            bmin = (vmax - vmin) * (idi + 0.0) / size + vmin
            bmax = (vmax - vmin) * (idi + 1.0) / size + vmin
            state[i] = np.random.uniform(bmin, bmax)
        return np.array(state)

def advance_dyn_model(dyn_model, state, act): # get new state using the learned dynamic model
    transitions = dyn_model[state][act]

    rnum = np.random.random()
    cum_prob = 0.0
    for k in transitions.keys():
        cum_prob += transitions[k][0]
        if rnum < cum_prob:
            return k, transitions[k][1]


def state_filter_cartpole(s):
    s[1] = s[1] % (2*np.pi)
    return s

def state_filter_hopper(s):
    return s[1:]