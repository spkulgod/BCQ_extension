import numpy as np
import matplotlib.pyplot as plt

file = 'BCQ_modified_gym_env:ReacherPyBulletEnv-v1_0'

rewards = np.load(file+'.npy', allow_pickle=True)
plt.plot(rewards)
plt.show()