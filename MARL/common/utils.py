import math
import torch as th
from torch.autograd import Variable
import numpy as np


def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1) * (log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot


def to_tensor_var(x, use_cuda=False, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))


def agg_list_stat(agg_list):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in agg_list]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    s_max = np.max(np.array(s), 0)
    s_min = np.min(np.array(s), 0)
    return s_mu, s_std, s_max, s_min


def exponential_epsilon_decay(epsilon_start, epsilon_end, decay_rate, episode):
    """
    Exponential epsilon decay function.

    Args:
        epsilon_start (float): Starting value of epsilon.
        epsilon_end (float): Minimum value of epsilon.
        decay_rate (float): Decay rate of epsilon e.g 0.01, 0.001, 0.0001.
        episode (int): Current episode number.

    Returns:
        float: Decayed epsilon value for the given episode.
    """
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-decay_rate * episode)
    return epsilon


def greedy_epsilon_decay(epsilon, epsilon_end, decay_rate):
    """
    Decrease the epsilon over time
        Args:
        epsilon (float): Current value of epsilon.
        epsilon_end (float): Minimum value of epsilon.
        decay_rate (float): Decay rate of epsilon e.g 0.99 0.995 0.98.

    Returns:
        float: Decayed epsilon value.
    """
    if epsilon > epsilon_end:
        epsilon *= decay_rate
        return max(epsilon_end, epsilon)
    return epsilon
