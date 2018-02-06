from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np


def estimate_task_grads(grad_fn, task_datasets):
    task_grads = []
    for t in task_datasets:
        task_grads.append(grad_fn(task_datasets[t]))
    return task_grads
