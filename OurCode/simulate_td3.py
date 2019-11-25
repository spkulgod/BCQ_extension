import numpy as np
import torch
import time
import gym
import matplotlib.pyplot as plt
import os

from TD3 import Actor

env_name = "ReacherPyBulletEnv-v0"

env = gym.make(env_name)
env.render()

model = env_name+'/actor_td3_tmp.pt'


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
		env.render()
		time.sleep(0.01)
	print(num)
