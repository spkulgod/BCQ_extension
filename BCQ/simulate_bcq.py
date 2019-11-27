import numpy as np
import torch
import time
import gym
import matplotlib.pyplot as plt
import os

import pybullet
import pybulletgym.envs

from BCQ import BCQ

# env_name = 'ReacherPyBulletEnv-v0'
env_name = 'Reacher-v2'
# env_name = "modified_gym_env:ReacherPyBulletEnv-v1"
env = gym.make(env_name)
env.render()

folder = 'results/'+env_name+'/'

bcq = BCQ(env.reset().shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
bcq.actor.load_state_dict(torch.load(folder+'bcq_actor.pt', map_location=torch.device('cpu')))
bcq.actor.eval()
bcq.critic.load_state_dict(torch.load(folder+'bcq_critic.pt', map_location=torch.device('cpu')))
bcq.critic.eval()
bcq.vae.load_state_dict(torch.load(folder+'bcq_vae.pt', map_location=torch.device('cpu')))
bcq.vae.eval()

model = folder+'bcq_actor.pt'
last_update_time = os.stat(model)[8]

while True:
	cur_time = os.stat(model)[8]
	if cur_time != last_update_time:
		bcq.actor.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
		bcq.actor.eval()
		last_update_time = cur_time

	time.sleep(0.5)
	done = False
	state = env.reset()
	num = 0
	while not done:
		num += 1
		state = torch.from_numpy(state).type(torch.FloatTensor)
		action = bcq.select_action(state)
		state, reward, done, _ = env.step(action)
		env.render()
		time.sleep(0.01)
	print(num)
