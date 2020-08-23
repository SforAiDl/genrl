import sys
sys.path.append("/home/aditya/Desktop/genrl/genrl") #add path
import numpy as np

from contrib.maa2c.a2c_agent import A2CAgent
from simple_spread_test import make_env
# from genrl.deep.common.trainer import OnPolicyTrainer
# from genrl.environments import VectorEnv

from utils import TensorboardLogger
# from genrl.deep.common.logger import TensorboardLogger


def train(max_episodes, timesteps_per_eps, logdir):

	# CUSTOM
	tensorboard_logger = TensorboardLogger(logdir)
	# vec_env = VectorEnv(
 #        env_id="simple_spread", n_envs=1, parallel=False, env_type="multi"
 #        )

	environment = make_env(scenario_name="simple_spread",benchmark=False)

	a2c_agent = A2CAgent(environment, 2e-4, 0.99, None, 0.008, timesteps_per_eps)

	for episode in range(1,max_episodes+1):

		print("EPISODE",episode)

		states = np.asarray(environment.reset())

		values, dones = a2c_agent.collect_rollouts(states)

		a2c_agent.get_traj_loss(values,dones)

		a2c_agent.update_policy()

		params = a2c_agent.get_logging_params()
		params["episode"] = episode

		tensorboard_logger.write(params)



	tensorboard_logger.close()

	# on_policy_trainer = OnPolicyTrainer(agent=a2c_agent,env=vec_env, log_mode=["tensorboard"], log_key="Episode" , save_interval=100, save_model="checkpoints", steps_per_epoch=timesteps_per_eps, epochs=max_episodes,
	# 	log_interval=1, batch_size=1)

	# on_policy_trainer.train()





if __name__ == '__main__':

	train(max_episodes = 10000, timesteps_per_eps = 300, logdir="./runs/")
