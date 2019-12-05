import numpy as np
import torch
import time
import gym
import matplotlib.pyplot as plt
import os

import pybullet
import pybulletgym.envs

from BCQ_modified import BCQ

# env_name = 'ReacherPyBulletEnv-v0'
# env_name = 'Hopper-v2'
env_name = 'Reacher-v2'
# env_name = "modified_gym_env:ReacherPyBulletEnv-v1"
env = gym.make(env_name)
# env.render()

folder = 'results_modified/'+env_name+'lr/'
folder = 'results_modified/'+env_name+'_buffer_mod_p_mixed_0.8_final/'

bcq = BCQ(env.reset().shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
bcq.critic.load_state_dict(torch.load(folder+'bcq_mod_critic_tmp.pt', map_location=torch.device('cpu')))
bcq.critic.eval()
bcq.vae.load_state_dict(torch.load(folder+'bcq_mod_vae_tmp.pt', map_location=torch.device('cpu')))
bcq.vae.eval()

model = folder+'bcq_mod_vae_tmp.pt'
last_update_time = os.stat(model)[8]

while True:
	cur_time = os.stat(model)[8]
	if cur_time != last_update_time:
		bcq.critic.load_state_dict(torch.load(folder+'bcq_mod_critic_tmp.pt', map_location=torch.device('cpu')))
		bcq.critic.eval()
		bcq.vae.load_state_dict(torch.load(folder+'bcq_mod_vae_tmp.pt', map_location=torch.device('cpu')))
		bcq.vae.eval()
		last_update_time = cur_time

	# time.sleep(0.5)
	done = False
	state = env.reset()
	# env.render()
	# time.sleep(5)
	num = 0
	while not done:
		num += 1
		state = torch.from_numpy(state).type(torch.FloatTensor)
		action = bcq.select_action(state)
		state, reward, done, _ = env.step(action)
		# print(env.get_body_com("target"))
		env.render()
		# time.sleep(0.01)
	print(num, env.sim.data.qpos[0])
