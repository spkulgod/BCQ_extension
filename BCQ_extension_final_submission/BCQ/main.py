import gym
import numpy as np
import torch
import argparse
import os

import pybullet
import pybulletgym.envs

import time
import matplotlib.pyplot as plt

import utils
import DDPG
import BCQ


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
	avg_reward = 0.
	avg_epis_size = 0
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward
			avg_epis_size += 1

	avg_reward /= eval_episodes
	avg_epis_size /= eval_episodes

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f episode length %f" % (eval_episodes, avg_reward, avg_epis_size))
	print ("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	for env_name, buffer_name in [('Reacher-v2', "/buffer_mod_p_mixed_0.8")]:
		parser = argparse.ArgumentParser()
		parser.add_argument("--env_name", default=env_name)												# OpenAI gym environment name
		parser.add_argument("--seed", default=0, type=int)												# Sets Gym, PyTorch and Numpy seeds
		parser.add_argument("--buffer_type", default="Robust")											# Prepends name to filename.
		parser.add_argument("--eval_freq", default=1e3, type=float)										# How often (time steps) we evaluate
		parser.add_argument("--max_timesteps", default=3e5, type=float)									# Max time steps to run environment for
		args = parser.parse_args()

		file_name = "BCQ_%s_%s" % (args.env_name, str(args.seed))

		print ("---------------------------------------")
		print ("Settings:", file_name, args.env_name, buffer_name[1:])
		print ("---------------------------------------")

		env = gym.make(args.env_name)
		env.seed(args.seed)
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
		
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])

		buffer_name = args.env_name+buffer_name

		# Load buffer
		replay_buffer = utils.ReplayBuffer()
		replay_buffer.load(buffer_name)
	
		args.env_name = env_name+"_"+buffer_name.split('/')[1]+'_final'
		
		os.chdir('results')
		if not os.path.exists(args.env_name):
			os.makedirs('./'+args.env_name)
		os.chdir('../')

		# Initialize policy
		policy = BCQ.BCQ(state_dim, action_dim, max_action)

		evaluations = []

		episode_num = 0
		done = True 

		training_iters = 0
		while training_iters < args.max_timesteps: 
			start = time.time()
			pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))
			stop = time.time()

			evaluations.append(evaluate_policy(policy))
			np.save("./results/"+ args.env_name + '/'+ file_name + '_tmp', evaluations)

			torch.save(policy.actor.state_dict(), './results/' + args.env_name + '/bcq_actor_tmp.pt')
			torch.save(policy.critic.state_dict(), './results/' + args.env_name + '/bcq_critic_tmp.pt')
			torch.save(policy.vae.state_dict(), './results/' + args.env_name + '/bcq_vae_tmp.pt')
			
			plt.plot(evaluations)
			plt.xlabel('Iterations (x5000)')
			plt.ylabel('Average Reward')
			plt.title('BCQ - Average Reward vs Iterations')
			plt.savefig('./results/' + args.env_name + '/bcq_tmp.png')
			plt.close()

			training_iters += args.eval_freq
			print ("Training iterations: " + str(training_iters), "Time:", int(stop-start))

			torch.save(policy.actor_target.state_dict(), './results/' + args.env_name + '/bcq_actor_target_tmp.pt')
			torch.save(policy.critic_target.state_dict(), './results/' + args.env_name + '/bcq_critic_target_tmp.pt')

		np.save("./results/"+ args.env_name + '/'+ file_name, evaluations)

		torch.save(policy.actor.state_dict(), './results/' + args.env_name + '/bcq_actor.pt')
		torch.save(policy.critic.state_dict(), './results/' + args.env_name + '/bcq_critic.pt')
		torch.save(policy.vae.state_dict(), './results/' + args.env_name + '/bcq_vae.pt')
		
		plt.plot(evaluations)
		plt.xlabel('Iterations (x5000)')
		plt.ylabel('Average Reward')
		plt.title('BCQ - Average Reward vs Iterations')
		plt.savefig('./results/' + args.env_name + '/bcq.png')
		plt.close()

		torch.save(policy.actor_target.state_dict(), './results/' + args.env_name + '/bcq_actor_target.pt')
		torch.save(policy.critic_target.state_dict(), './results/' + args.env_name + '/bcq_critic_target.pt')
