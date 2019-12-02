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

env_name = "Reacher-v2"
model = env_name+'/actor_td3.pt'

env = gym.make(env_name)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

actor = Actor(env.reset().shape[0], env.action_space.shape[0])
actor.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
actor.eval()

Buffer = Replay(1.2e6,0,env.reset().shape[0], env.action_space.shape[0],env)

done = False
state = env.reset()

min_thr = 0
prob_thr = 0.75
reverse = False
while len(Buffer.buffer) != 1e6:
    if done:
        state = env.reset()
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action = actor(state).cpu().detach().numpy()
    tip = env.get_body_com("fingertip")
    if (tip[0]>min_thr and tip[1]<min_thr):
        reverse = True
        p = np.random.uniform(0,1)
        if p > prob_thr:
            action = -action
            next_state, reward, done, _ = env.step(action)
            Buffer.buffer_add((state, next_state, action, reward, 1-done))
        else:
            next_state, reward, done, _ = env.step(action)
            if p> prob_thr-0.5:
                Buffer.buffer_add((state, next_state, action, reward, 1-done))
    else:
        next_state, reward, done, _ = env.step(action)
        Buffer.buffer_add((state, next_state, action, reward, 1-done))

    reverse = False
    state = next_state

    length = len(Buffer.buffer)
    if length % 5000 == 0:
        print(length)

np.save(env_name + '/buffer_mod_p_mixed_' + str(prob_thr) + '.npy', Buffer.buffer)