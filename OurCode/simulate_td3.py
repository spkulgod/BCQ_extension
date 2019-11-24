import numpy as np
import torch
import time
import gym
import matplotlib.pyplot as plt
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# from TD3 import Actor

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a)) 
		return a

env_name = "Hopper-v2"

env = gym.make(env_name)
env.render()

model = 'DDPG_Hopper-v2_0_actor.pth'

actor = Actor(env.reset().shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
actor.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
actor.eval()
last_update_time = os.stat(model)[8]

while True:
	cur_time = os.stat(model)[8]
	if cur_time != last_update_time:
		actor.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
		actor.eval()
		last_update_time = cur_time

	time.sleep(0.5)
	done = False
	state = env.reset()
	num = 0
	while not done:
		num += 1
		state = torch.from_numpy(state).type(torch.FloatTensor)
		action = actor(state)
		env.render()
		state, reward, done, _ = env.step(action.detach().numpy())
		time.sleep(0.01)
	print(num)
