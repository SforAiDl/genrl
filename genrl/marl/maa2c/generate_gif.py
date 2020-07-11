from simple_spread_test import make_env
from maa2c import MAA2C
import imageio
import numpy as np
import os
import torch
os.chdir("/home/aditya/Desktop/Partial_Reward_Decoupling/PRD/SimpleSpread/models_gif")


if __name__ == '__main__':
	env = make_env(scenario_name="simple_spread")
	ma_controller = MAA2C(env)




	# Number of images to capture
	n_images = 10000

	images = []

	# init a new episode
	obs = env.reset()
	# init the img var with the starting state of the env
	img = env.render(mode='rgb_array')[0]

	for i in range(n_images):
	  # At each step, append an image to list
	  images.append(img)

	  # Advance a step and render a new image
	  with torch.no_grad():
	    action = ma_controller.get_actions(obs)
	  obs, _, _ ,_ = env.step(action)
	  img = env.render(mode='rgb_array')[0]


	# print(images)

	imageio.mimwrite('./simple_spread.gif',
	                [np.array(img) for i, img in enumerate(images) if i%2 == 0],
	                fps=50)

	print("DONE!")

# import pyglet
# window = pyglet.window.Window()
# pyglet.app.run()
