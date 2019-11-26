""" Learn a policy using DDPG for the reach task"""
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
import random
from copy import deepcopy

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def weighSync(target_model, source_model, tau=0.001):
	# update target networks 
	for target_param, param in zip(target_model.parameters(), source_model.parameters()):
		target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

	   
class Replay():
	def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
		"""
		A function to initialize the replay buffer.

		param: init_length : Initial number of transitions to collect
		param: state_dim : Size of the state space
		param: action_dim : Size of the action space
		param: env : gym environment object
		"""
		self.buffer = []
		self.buffer_size = buffer_size
		self.random_initialise(env, init_length, action_dim)

	def random_initialise(self, env, init_length, action_dim):
		state = env.reset()
		done = False
		for _ in range(init_length):
			if done:
				state = env.reset()
			action = np.random.uniform(-1, 1, action_dim)
			next_state, reward, done, _ = env.step(action)
			self.buffer.append((state, next_state, action, reward,1-done))
			state = next_state

	def buffer_add(self, exp):
		"""
		A function to add a dictionary to the buffer
		param: exp : A dictionary consisting of state, action, reward , next state and not_done flag
		"""
		self.buffer.append(exp)
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)
		
	def buffer_sample(self, N):
		"""
		A function to sample N points from the buffer
		param: N : Number of samples to obtain from the buffer
		"""
		state = []
		action = []
		reward = []
		next_state = []
		not_done = []
		for s, ns, a, r, d in random.sample(self.buffer, N):
			state.append(s)
			action.append(a)
			reward.append(r)
			next_state.append(ns)
			not_done.append(d)
		return torch.FloatTensor(state).cuda(), torch.FloatTensor(action).cuda(),\
		 		torch.FloatTensor(reward).resize_((len(reward), 1)).cuda(),\
				torch.FloatTensor(next_state).cuda(),\
				torch.FloatTensor(not_done).resize_((len(not_done), 1)).cuda()

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		"""
		Initialize the network
		param: state_dim : Size of the state space
		param: action_dim: Size of the action space
		"""
		super(Actor, self).__init__()
		hl1 = 400
		hl2 = 300

		self.fc1 = nn.Linear(state_dim, hl1)
		self.fc2 = nn.Linear(hl1, hl2)
		self.fc3 = nn.Linear(hl2, action_dim)
		#self.fc1.weight.data.uniform_(-1/np.sqrt(state_dim), 1/np.sqrt(state_dim))
		#self.fc2.weight.data.uniform_(-1/np.sqrt(hl1), 1/np.sqrt(hl1))
		#self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		"""
		Define the forward pass
		param: state: The state of the environment
		"""
		state = F.relu(self.fc1(state))
		state = F.relu(self.fc2(state))
		state = torch.tanh(self.fc3(state))
		return state

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		"""
		Initialize the critic
		param: state_dim : Size of the state space
		param: action_dim : Size of the action space
		"""
		super(Critic, self).__init__()
		hl1 = 400
		hl2 = 300

		self.fc1 = nn.Linear(state_dim, hl1)
		self.fc2 = nn.Linear(hl1 + action_dim, hl2)
		self.fc3 = nn.Linear(hl2, 1)
		# self.fc1.weight.data.uniform_(-1/np.sqrt(state_dim), 1/np.sqrt(state_dim))
		# self.fc2.weight.data.uniform_(-1/np.sqrt(hl1 + action_dim), 1/np.sqrt(hl1 + action_dim))
		# self.fc3.weight.data.uniform_(-3e-3, 3e-3)
		
	def forward(self, state, action):
		"""
		Define the forward pass of the critic
		"""
		state = F.relu(self.fc1(state))
		state = F.relu(self.fc2(torch.cat((state, action), 1)))
		state = self.fc3(state)
		return state

class DDPG():
	def __init__(
			self,
			env,
			action_dim,
			state_dim,
			critic_lr=1e-3,
			actor_lr=1e-4,
			gamma=0.99,
			batch_size=64,
	):
		"""
		param: env: An gym environment
		param: action_dim: Size of action space
		param: state_dim: Size of state space
		param: critic_lr: Learning rate of the critic
		param: actor_lr: Learning rate of the actor
		param: gamma: The discount factor
		param: batch_size: The batch size for training
		"""
		self.gamma = gamma
		self.batch_size = batch_size
		self.env = env
		self.eval_env = deepcopy(env)

		self.action_dim = action_dim
		self.state_dim = state_dim

		self.actor = Actor(state_dim, action_dim).cuda()
		self.actor_target = Actor(state_dim, action_dim).cuda()

		self.critic = Critic(state_dim, action_dim).cuda()
		self.critic_target = Critic(state_dim, action_dim).cuda()

		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data)

		self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay = 1e-2)

		self.ReplayBuffer = Replay(1000000, 1000, state_dim, action_dim, self.env)

	def update_target_networks(self):
		"""
		A function to update the target networks
		"""
		weighSync(self.actor_target, self.actor, 0.001)
		weighSync(self.critic_target, self.critic, 0.001)

	def evaluate(self):
		state = self.eval_env.reset()
		done = False
		returns = 0
		t = 0
		while not done:
			action = self.actor(torch.from_numpy(state).type(torch.FloatTensor).cuda())
			next_state, reward, done, _ = self.eval_env.step(action.cpu().detach().numpy())
			state = next_state
			returns += self.gamma ** t * reward
			t += 1
		return returns


	def train(self, num_steps):
		"""
		Train the policy for the given number of iterations
		:param num_steps:The number of steps to train the policy for
		"""
		MSELoss = nn.MSELoss()
		return_store = []
		start = time.time()
		done = False
		state = self.env.reset()
		for i in range(num_steps):
			if done:
				state = self.env.reset()
			
			action = self.actor(torch.from_numpy(state).type(torch.FloatTensor).cuda()).cpu().detach().numpy()+np.random.normal(0, np.sqrt(0.1), size=self.action_dim)
			next_state, reward, done, _ = self.env.step(action)

			self.ReplayBuffer.buffer_add((state, next_state, action, reward, 1-done))
			state = next_state
			
			states, actions, rewards, next_states, not_dones = self.ReplayBuffer.buffer_sample(self.batch_size)
			y = rewards + not_dones*self.gamma * self.critic_target(next_states, self.actor_target(next_states).detach()).detach()
			Q = self.critic(states, actions)

			self.optimizer_critic.zero_grad()
			loss_critic = MSELoss(y, Q)
			loss_critic.backward()
			self.optimizer_critic.step()

			self.optimizer_actor.zero_grad()
			loss_actor = -self.critic(states, self.actor(states)).mean()
			loss_actor.backward()
			self.optimizer_actor.step()

			if i%100 == 0:
				return_store.append(self.evaluate())

				if i % 5000 == 0:
					stop = time.time()
					print(i, "Time: ", int(stop-start))
					torch.save(self.actor.state_dict(), env_name+'/actor_ddpg_tmp.pt')
					np.save(env_name+'/returns_ddpg_tmp.npy', return_store)
					start = stop

					plt.plot(return_store)
					plt.xlabel('Iterations (x100)')
					plt.ylabel('Discounted Returns')
					plt.title('DDPG - Discounted Returns vs Iterations')
					plt.savefig(env_name+'/ddpg_tmp.png')
					plt.close()

					if i % 25000 == 0:
						np.save(env_name+'/buffer_ddpg_tmp.npy', self.ReplayBuffer.buffer) 

			self.update_target_networks()

		torch.save(self.actor.state_dict(), env_name+'/actor_ddpg.pt')
		np.save(env_name+'/returns_ddpg.npy', return_store)
		np.save(env_name+'/buffer_ddpg.npy', self.ReplayBuffer.buffer)
		torch.save(self.actor_target.state_dict(), env_name+'/actor_target_ddpg.pt')

		plt.plot(return_store)
		plt.xlabel('Iterations (x100)')
		plt.ylabel('Discounted Returns')
		plt.title('DDPG - Discounted Returns vs Iterations')
		plt.savefig(env_name+'/ddpg.png')
		plt.close()
		# plt.show()


if __name__ == "__main__":
	# Define the environment
	env_name = "Hopper-v2"
	if not os.path.exists("./"+env_name):
		os.makedirs("./"+env_name)
	env = gym.make(env_name)
	env.seed(seed)

	observation = env.reset()
	ddpg_object = DDPG(
		env,
		env.action_space.shape[0],
		observation.shape[0],
		critic_lr=1e-3,
		actor_lr=1e-4,
		gamma=0.99,
		batch_size=256,
	)
 
	# Train the policy
	ddpg_object.train(400000)

	# Evaluate the final policy
	state = env.reset()

	done = False
	while not done:
		action = ddpg_object.actor(torch.from_numpy(state).type(torch.FloatTensor).cuda())
		next_state, reward, done, _ = env.step(action.cpu().detach().numpy())
		time.sleep(0.1)
		state = next_state
