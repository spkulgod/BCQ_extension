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

import random
from copy import deepcopy

from TD3 import Actor, Replay

env_name = "Reacher-v2"

env = gym.make(env_name)
model = env_name+'/actor_td3_tmp.pt'

actor = Actor(env.reset().shape[0], env.action_space.shape[0])
actor.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
actor.eval()

buffer = Replay(1e6,0,env.reset().shape[0], env.action_space.shape[0],env)

done = False
state = env.reset()

########################3
ind = 0
min = 0
max = 0

for _ in range(int(buffer.buffer_size)):
    if done:
        state = env.reset()
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action = actor(state).cpu().detach().numpy()
    tip = env.get_body_com("fingertip")
    if (tip[0]>min and tip[1]<min):
        vec = tip-env.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        print("init: ", state)
        env.do_simulation(-action, env.frame_skip)
        next_state = env.env._get_obs()
        print("mid: ", next_state)
        buffer.buffer_add((state, next_state, -action, reward, 1-done))
        env.do_simulation(action, env.frame_skip)
        print("final: ", env.env._get_obs(),"\n")
        next_state, reward, done, _ = env.step(action)
    else:
        next_state, reward, done, _ = env.step(action)
        buffer.buffer_add((state, next_state, action, reward, 1-done))
    state = next_state

np.save(env_name+'/buffer_mod.npy', buffer.buffer)
