import math
import cv2, os
import torch as th
from torch.autograd import Variable
import numpy as np
from shutil import copy
import torch.nn as nn


def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1)*(log_p2 - log_p1), 1)


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


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    s_max = np.max(np.array(s), 0)
    s_min = np.min(np.array(s), 0)
    return s_mu, s_std, s_max, s_min


class VideoRecorder:
    """This is used to record videos of evaluations"""

    def __init__(self, filename, frame_size, fps):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), int(fps),
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


def init_dir(base_dir, pathes=['train_videos', 'configs', 'models', 'eval_videos', 'eval_logs']):
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def exponential_epsilon_decay(epsilon_start, epsilon_end, decay_rate, episode):
    """
    Exponential epsilon decay function.

    Args:
        epsilon_start (float): Starting value of epsilon.
        epsilon_end (float): Minimum value of epsilon.
        decay_rate (float): Decay rate of epsilon.
        episode (int): Current episode number.

    Returns:
        float: Decayed epsilon value for the given episode.
    """
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-decay_rate * episode)
    return epsilon