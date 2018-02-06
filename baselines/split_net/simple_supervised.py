import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.common.mpi_adam import MpiAdam
import copy
from baselines.split_net.mlp_net import *
import matplotlib.pyplot as plt


def train(weight1, weight2, shared_bits, Xtrain, Ytrain, task1num, Xtest, Ytest, task1testnum, learning_rate, iternum, perturb=0):
    grad_stds = []
    loss = []
    for i in range(iternum):
        train_loss1 = test(Xtrain[0:task1num], Ytrain[0:task1num], weight1)
        train_loss2 = test(Xtrain[task1num:], Ytrain[task1num:], weight2)
        grad1 = 1.0 * 1.0 / len(Xtrain[0:task1num]) * np.dot(
            2 * (np.dot(Xtrain[0:task1num], weight1) - Ytrain[0:task1num]), Xtrain[0:task1num])
        grad2 = 1.0 * 1.0 / len(Xtrain[task1num:]) * np.dot(
            2 * (np.dot(Xtrain[task1num:], weight2) - Ytrain[task1num:]),
            Xtrain[task1num:])
        grad1final = grad1 * (1 - shared_bits) + (grad1 + grad2) * 0.5 * shared_bits
        grad2final = grad2 * (1 - shared_bits) + (grad1 + grad2) * 0.5 * shared_bits

        totalg = 0.5 * (grad1 + grad2)
        exp_perf = np.abs(grad1) + np.abs(grad2) - np.abs(totalg)
        grad_stds.append(exp_perf)

        weight1 -= learning_rate * grad1final
        weight2 -= learning_rate * grad2final
        loss.append(0.5 * (test(Xtest[0:task1testnum], Ytest[0:task1testnum], weight1) + test(Xtest[task1testnum:], Ytest[task1testnum:], weight2)))

    return weight1, weight2, grad_stds, loss


def sample_data(dim, num, task, diff):
    Xs = []
    Ys = []

    xscale = np.array(np.arange(dim), dtype=np.float32) + 1
    #if task == 1:
    #    xscale -= 4.0
    if task == 0:
        true_weights = np.random.random(dim) * 4
        #true_weights[2] = 4
        #true_weights[8] = 4
    else:
        true_weights = np.random.random(dim) * 4
        #true_weights[2] = 4
        #true_weights[8] = 4
    if diff:
        true_weights[-1] = 0
    for i in range(num):
        x = xscale * np.random.random(dim)
        Ys.append(np.dot(true_weights, x))
        if diff:
            x[-1] = 0
        Xs.append(x)

    return Xs, Ys, true_weights

def test(X, Y, weight):
    return np.mean((np.dot(X, weight) - Y) ** 2)

def main():
    np.random.seed(17)

    dim = 16
    num = 5000

    Xs1, Ys1, true1 = sample_data(dim, num, 0, False)
    Xs2, Ys2, true2 = sample_data(dim, num, 1, False)

    Xtrain = Xs1[0:int(num * 0.8)] + Xs2[0:int(num * 0.8)]
    Ytrain = Ys1[0:int(num * 0.8)] + Ys2[0:int(num * 0.8)]
    Xtest = Xs1[int(num * 0.8):] + Xs2[int(num * 0.8):]
    Ytest = Ys1[int(num * 0.8):] + Ys2[int(num * 0.8):]

    weight1 = np.random.random(dim)
    weight2 = np.copy(weight1)

    learning_rate = 0.001

    loss1 = []
    loss2 = []
    grad_stds = []
    task1num = len(Xtrain)//2
    task1testnum = len(Xtest)//2
    shared_bits = np.ones(dim)
    performance_vs_sharing = []

    # joint training
    shared_bits = np.ones(dim)
    weight1, weight2, _, loss = train(weight1, weight2, shared_bits, Xtrain, Ytrain, task1num, Xtest, Ytest,
                                      task1testnum, learning_rate, 100)
    init_weight = np.copy(weight1)

    for share in range(dim):
        shared_bits = np.ones(dim)
        shared_bits[share] = 0.0
        weight1 = np.copy(init_weight)
        weight2 = np.copy(weight1)
        weight1, weight2, _, loss = train(weight1, weight2, shared_bits, Xtrain, Ytrain, task1num, Xtest, Ytest, task1testnum, learning_rate, 100)

        print('End loss: ', share,  loss[-1])
        performance_vs_sharing.append(loss[-1])

    init_weight = np.copy(weight1)
    weight1 = np.copy(init_weight)
    weight2 = np.copy(weight1)
    shared_bits = np.ones(dim)
    weight1, weight2, grad_stds, loss = train(weight1, weight2, shared_bits, Xtrain, Ytrain, task1num, Xtest, Ytest,
                                                  task1testnum, learning_rate, 5, perturb=5)
    exp_perfs = []
    for g1, g2 in grad_stds:
        totalg = 0.5 * (g1+g2)
        exp_perf = np.abs(g1) + np.abs(g2) - np.abs(totalg)
        exp_perfs.append(np.std([g1,g2], axis=0))
    print('================')
    print('perf ', np.argsort(performance_vs_sharing))
    print('est ', np.argsort(-np.mean(exp_perfs, axis=0)))

    plt.figure()
    plt.title('exp_perfs')
    plt.plot(np.mean(exp_perfs, axis=0), label='exp_perfs')
    plt.legend()

    plt.figure()
    plt.title('performance_vs_sharing')
    plt.plot(performance_vs_sharing, label='performance_vs_sharing')
    plt.legend()

    plt.figure()
    plt.title('weights')
    plt.plot(true1, label='true1')
    plt.plot(true2, label='true2')
    plt.plot(weight1, label='learn1')
    plt.plot(weight2, label='learn2')
    plt.legend()

    '''
    # split all training
    shared_bits = np.zeros(dim)
    weight1 = np.copy(init_weight)
    weight2 = np.copy(weight1)
    sep_train_loss = []
    weight1, weight2, grad_stds, loss = train(weight1, weight2, shared_bits, Xtrain, Ytrain, task1num, Xtest, Ytest,
                                  task1testnum, learning_rate, 100)
    sep_train_loss += loss

    shared_bits = np.ones(dim)
    weight1 = np.copy(init_weight)
    weight2 = np.copy(weight1)
    cont_split_loss = []
    for i in range(20):
        weight1, weight2, grad_stds, loss = train(weight1, weight2, shared_bits, Xtrain, Ytrain, task1num, Xtest, Ytest,
                                                  task1testnum, learning_rate, 5)
        grad_sort = np.argsort(-np.mean(grad_stds, axis=0))
        shared_bits[grad_sort[[0,1]]] = 0.0
        cont_split_loss += loss


    plt.figure()
    plt.title('weights')
    plt.plot(true1, label='true1')
    plt.plot(true2, label='true2')
    plt.plot(weight1, label='learned 1')
    plt.plot(weight2, label='learned 2')
    plt.legend()

    plt.figure()
    plt.title('loss')
    plt.plot(sep_train_loss, label='loss seperate training')
    plt.plot(cont_split_loss, label='continuous splitting')
    plt.legend()
    '''

    plt.show()


if __name__ == '__main__':
    main()






