import numpy as np
import torch
import time
import gym
import matplotlib.pyplot as plt
import os

import pybullet
import pybulletgym.envs

from BCQ_modified import BCQ

# env_name = 'Hopper-v2'
env_name = 'Reacher-v2'
env = gym.make(env_name)

folder = 'results_modified/'+env_name+'/'

bcq = BCQ(env.reset().shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
bcq.critic.load_state_dict(torch.load(folder+'bcq_mod_critic_tmp.pt', map_location=torch.device('cpu')))
bcq.critic.eval()
bcq.vae.load_state_dict(torch.load(folder+'bcq_mod_vae_tmp.pt', map_location=torch.device('cpu')))
bcq.vae.eval()

model = folder+'bcq_mod_vae_tmp.pt'
last_update_time = os.stat(model)[8]

env.reset()
while True:
	cur_time = os.stat(model)[8]
	if cur_time != last_update_time:
		bcq.critic.load_state_dict(torch.load(folder+'bcq_mod_critic_tmp.pt', map_location=torch.device('cpu')))
		bcq.critic.eval()
		bcq.vae.load_state_dict(torch.load(folder+'bcq_mod_vae_tmp.pt', map_location=torch.device('cpu')))
		bcq.vae.eval()
		last_update_time = cur_time

	done = False
	state = env.reset()
	while not done:
		state = torch.from_numpy(state).type(torch.FloatTensor)
		action = bcq.select_action(state)
		state, reward, done, _ = env.step(action)
		env.render()
