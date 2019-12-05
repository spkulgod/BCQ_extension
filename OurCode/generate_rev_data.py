import numpy as np
import torch
import time
import torch.nn as nn
import os

import gym
import pybullet
import pybulletgym.envs

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from TD3 import Actor, Replay


def check(env_name):
    if env_name == 'Reacher-v2':
        thr = 0
        val = env.get_body_com("fingertip")
        return True or val[0]>thr and val[1]<thr

    elif env_name == 'Hopper-v2':
        min_thr = 0.2
        max_thr = 2.2
        val = env.sim.data.qpos[0]
        return val < max_thr and val > min_thr

    else:
        1/0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = "Reacher-v2"
model = env_name+'/actor_td3.pt'

env = gym.make(env_name)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

actor = Actor(env.reset().shape[0], env.action_space.shape[0])
actor.load_state_dict(torch.load(model, map_location=torch.device("cpu")))
actor.eval()

Buffer = Replay(4e5,0,env.reset().shape[0], env.action_space.shape[0],env)

done = False
state = env.reset()

prob_thr = 0.9
reverse = False
number_correct = 0
correct_percentage = 0.8
while len(Buffer.buffer) != Buffer.buffer_size:
    if done:
        state = env.reset()
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action = actor(state).cpu().detach().numpy()

    if check(env_name):
        reverse = True
        p = np.random.uniform(0,1)
        if p > prob_thr:
            action = -action
            next_state, reward, done, _ = env.step(action)
            Buffer.buffer_add((state, next_state, action, reward, 1-done))
        else:
            next_state, reward, done, _ = env.step(action)
            if number_correct / Buffer.buffer_size < correct_percentage:
                Buffer.buffer_add((state, next_state, action, reward, 1-done))
                number_correct += 1
            else:
                prob_thr = 0.8
    elif number_correct / Buffer.buffer_size < correct_percentage:
        next_state, reward, done, _ = env.step(action)
        Buffer.buffer_add((state, next_state, action, reward, 1-done))
        number_correct += 1
    else:
        next_state, reward, done, _ = env.step(action)

    reverse = False
    state = next_state

    length = len(Buffer.buffer)
    if length % 1000 == 0:
        print(length)

np.save(env_name + '/buffer_mod_p_mixed_' + str(prob_thr) + '.npy', Buffer.buffer)