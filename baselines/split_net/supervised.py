import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.common.mpi_adam import MpiAdam
import copy
from baselines.split_net.mlp_net import *

def train(loss_grad, optimizer, X, Y, iter, stepsize, batch = 32, shuffle=True):
    X=np.array(X)
    Y=np.array(Y)
    losses = []
    for epoch in range(iter):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(X) - batch + 1, batch):
            excerpt = indices[start_idx:start_idx + batch]
            input = np.array(X[excerpt])
            output = np.array(Y[excerpt])
            loss, g = loss_grad(input, output)
            train_err += loss[0]
            optimizer.update(g, stepsize)
            train_batches += 1
        losses.append(train_err / train_batches)
        #print("aux training loss:\t\t{:.6f}".format(train_err / train_batches))
    return losses

# default task is to reverse the order
# difficulty [0, dim-1] is the number of operations being mutated
def sample_tasks(dim, difficulties, seed = None):
    if seed is not None:
        np.random.seed(seed)
    tasks = []
    default_task = []
    for i in range(dim):
        default_task.append(dim-1-i)
    tasks.append(default_task)
    for difficulty in difficulties:
        default = copy.deepcopy(default_task)
        unmutated_lsit = np.arange(dim).tolist()
        for mutation in range(int(difficulty)):
            mutate_target = unmutated_lsit[np.random.randint(len(unmutated_lsit))]
            unmutated_lsit.remove(mutate_target)
            idx1 = np.random.randint(dim)
            idx2 = idx1
            while idx2 == idx1:
                idx2 = np.random.randint(dim)
            default[mutate_target] = idx1
        tasks.append(default)
    return tasks


def synthesize_data(dim, size, tasks, split = False, seed = None):
    # synthetic task of shuffling and calculating
    if seed is not None:
        np.random.seed(seed)
    Xs = []
    Ys = []

    per_task_size = [size / len(tasks)] * len(tasks)

    for i, task in enumerate(tasks):
        X=[]
        Y=[]
        for _ in range(int(per_task_size[i])):
            input = np.random.uniform(-10, 10, dim)
            input = np.concatenate([input, [i*1.0/(len(tasks)-1)]])
            if split:
                split_vec = [0] * len(tasks)
                split_vec[i] = 1
                input = np.concatenate([input, split_vec])
            output = np.zeros(dim)
            exec_task = copy.deepcopy(tasks[i])
            for idx, subtask in enumerate(exec_task):
                output[idx] = input[subtask]
            X.append(input)
            Y.append(output/10.0)
        Xs.append(X)
        Ys.append(Y)
    return Xs, Ys

def test(model, X, Y):
    pred = model.pred(X)
    return np.mean((pred-Y)**2)

def main():
    dim = 16
    difficulties = [8, 8]

    task_specs = sample_tasks(dim, difficulties, seed=0)

    Xs, Ys = synthesize_data(dim, 10000, task_specs, )

    # define joint model
    model_joint = MlpNet('joint_model', dim, dim, 64, 2)
    adam_joint = MpiAdam(model_joint.get_trainable_variables(), epsilon=1e-5)

    input = U.get_placeholder_cached(name="input")
    output = U.get_placeholder(name="label", dtype=tf.float32, shape=[None, dim])

    loss = U.mean((model_joint.mean - output)**2)
    grad = tf.gradient(loss, model_joint.get_trainable_variables())
















