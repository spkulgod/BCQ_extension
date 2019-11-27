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
import BCQ_modified


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
	
	parser = argparse.ArgumentParser()
	env_name = "Hopper-v2"
	# env_name = "Reacher-v2"
	# env_name = "ReacherPyBulletEnv-v0"
	# env_name = "modified_gym_env:ReacherPyBulletEnv-v1"
	parser.add_argument("--env_name", default=env_name)												# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)												# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="Robust")											# Prepends name to filename.
	parser.add_argument("--eval_freq", default=5e3, type=float)										# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)									# Max time steps to run environment for
	args = parser.parse_args()


	file_name = "BCQ_%s_%s" % (args.env_name, str(args.seed))
	# buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
	buffer_name = args.env_name+"/buffer_td3"

	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("---------------------------------------")

	env = gym.make(args.env_name)
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	args.env_name = args.env_name + '_target'

	os.chdir('results_modified')
	if not os.path.exists(args.env_name):
		os.makedirs('./'+args.env_name)
	os.chdir('../')
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Load buffer
	replay_buffer = utils.ReplayBuffer()
	replay_buffer.load(buffer_name)
		
	# for k, max_timesteps in [(0.5, 3e5)]:
	k, max_timesteps = (0.5, 3e5)
	for lr_critic, lr_vae in [(1e-3, 1e-3)]:
	# for lr_critic, lr_vae in [(1e-3, 1e-4), (1e-4, 1e-2), (1e-2, 1e-4), (1e-4, 1e-4)]:
	# for lr_critic, lr_vae in [(1e-2, 1e-2), (1e-2, 1e-3), (1e-3, 1e-2), (1e-4, 1e-3)]:
		args.max_timesteps = max_timesteps
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print('lr_critic =', lr_critic, 'lr_vae', lr_vae)
		print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

		# Initialize policy
		policy = BCQ_modified.BCQ(state_dim, action_dim, max_action)

		evaluations = []

		########## FOR STARTING FROM PREVIOUS RUN ###############
		# folder = 'results_modified/'+args.env_name+'/'
		# policy.critic.load_state_dict(torch.load(folder+'bcq_mod_critic_tmp1.pt'))
		# policy.critic_target.load_state_dict(torch.load(folder+'bcq_mod_critic_target_tmp1.pt'))
		# policy.vae.load_state_dict(torch.load(folder+'bcq_mod_vae_tmp1.pt'))
		# evaluations = np.load(folder+file_name+'_mod_tmp1.npy').tolist()
		########## FOR STARTING FROM PREVIOUS RUN ###############

		episode_num = 0
		done = True 

		training_iters = 0
		while training_iters < args.max_timesteps: 
			start = time.time()
			pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), k=k)
			stop = time.time()

			evaluations.append(evaluate_policy(policy))
			np.save("./results_modified/"+ args.env_name + '/'+ file_name + '_mod_tmp', evaluations)

			torch.save(policy.critic.state_dict(), './results_modified/' + args.env_name + '/bcq_mod_critic_tmp.pt')
			torch.save(policy.vae.state_dict(), './results_modified/' + args.env_name + '/bcq_mod_vae_tmp.pt')
			
			plt.plot(evaluations)
			plt.xlabel('Iterations (x5000)')
			plt.ylabel('Average Reward')
			plt.title('BCQ - Average Reward vs Iterations')
			plt.savefig('./results_modified/' + args.env_name + '/bcq_mod_tmp.png')
			plt.close()

			training_iters += args.eval_freq
			print ("Training iterations: " + str(training_iters), "Time:", int(stop-start))

			torch.save(policy.critic_target.state_dict(), './results_modified/' + args.env_name + '/bcq_mod_critic_target_tmp.pt')

		np.save("./results_modified/"+ args.env_name + '/'+ file_name + '_lr_cri_' + str(lr_critic) + '_lr_vae_' + str(lr_vae), evaluations)

		torch.save(policy.critic.state_dict(), './results_modified/' + args.env_name + '/bcq_mod_critic_lr_cri_' + str(lr_critic) + '_lr_vae_' + str(lr_vae) +'.pt')
		torch.save(policy.vae.state_dict(), './results_modified/' + args.env_name + '/bcq_mod_vae_lr_cri_' + str(lr_critic) + '_lr_vae_' + str(lr_vae) + '.pt')
		
		plt.plot(evaluations)
		plt.xlabel('Iterations (x5000)')
		plt.ylabel('Average Reward')
		plt.title('BCQ - Average Reward vs Iterations')
		plt.savefig('./results_modified/' + args.env_name + '/bcq_mod_lr_cri_' + str(lr_critic) + '_lr_vae_' + str(lr_vae) +'.png')
		plt.close()

		torch.save(policy.critic_target.state_dict(), './results_modified/' + args.env_name + '/bcq_mod_critic_target_lr_cri_' + str(lr_critic) + '_lr_vae_' + str(lr_vae) + '.pt')
