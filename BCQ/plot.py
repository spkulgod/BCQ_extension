import numpy as np
import matplotlib.pyplot as plt
import os

def running_mean(array, N):
    cumsum = np.cumsum(np.insert(array, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

seed = 0
lr_critic, lr_vae = (1e-3, 1e-3)

num = 0
char = ['Fig 1', 'Fig 2', 'Fig 3', 'Fig 4']
for env_name, buffer_name in [('Reacher-v2', "/buffer_td3"), ('Reacher-v2', "/buffer_mod_p_mixed_0.75"), ('Hopper-v2', "/buffer_td3"), ('Hopper-v2', "/buffer_mod_p_mixed_0.6")]:
	file_name = "BCQ_%s_%s" % (env_name, str(seed))
	env_name = env_name+"_"+buffer_name.split('/')[1]+'_final'

	os.chdir('../results_combined')
	if not os.path.exists(env_name):
		os.makedirs('./'+env_name)
	os.chdir('../BCQ/')

	file_mod = "./results_modified/"+ env_name + '/'+ file_name + '_lr_cri_' + str(lr_critic) + '_lr_vae_' + str(lr_vae)
	file_bcq = "./results/"+ env_name + '/'+ file_name

	rewards_mod = np.load(file_mod+'.npy', allow_pickle=True)
	rewards_bcq = np.load(file_bcq+'.npy', allow_pickle=True)

	plt.plot(running_mean(rewards_mod, 10), label='Improved BCQ')
	plt.plot(running_mean(rewards_bcq, 10), label='BCQ')
	plt.plot(rewards_mod, alpha=0.4)
	plt.plot(rewards_bcq, alpha=0.15)
	plt.xlabel('Iterations (x1000)')
	plt.ylabel('Average Reward')
	if buffer_name == "/buffer_td3":
		plt.title(char[num] + ': ' + env_name.split('_')[0] + ', Concurrent Batch')
	else:
		plt.title(char[num] + ': ' + env_name.split('_')[0] + ', Reversed Batch')
	plt.legend()
	# plt.show()
	plt.savefig('../results_combined/' + env_name + '/' + env_name + '_mean.png')
	plt.close()
	num += 1
