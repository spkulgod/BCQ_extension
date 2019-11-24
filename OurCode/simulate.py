import numpy as np
import torch
import time
import gym
import matplotlib.pyplot as plt
import os

from TD3 import Actor

env = gym.make('modified_gym_env:ReacherPyBulletEnv-v1')
env.render()

model = 'actor_td3.pt'


actor = Actor(env.reset().shape[0], env.action_space.shape[0])
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
		state, reward, done, _ = env.step(action.detach().numpy())
		time.sleep(0.01)
	print(num)
