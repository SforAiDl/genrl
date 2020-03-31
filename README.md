# JigglypuffRL
[![pypi](https://img.shields.io/badge/pypi-jigglypuff--rl-blue)](https://pypi.org/project/jigglypuff-rl/)
[![GitHub license](https://img.shields.io/github/license/SforAiDl/JigglypuffRL)](https://github.com/SforAiDl/JigglypuffRL/blob/master/LICENSE)

## Installation
We suggest creating a conda virtual environment before installing our package.
```
conda create -n JgRL_env python=3.6 pip
conda activate JgRL_env

pip install jigglypuff-rl
```

## Usage
```python
from jigglypuffRL import PPO1
import gym

env = gym.make('CartPole-v0')
agent = PPO1('MlpPolicy', 'MlpValue', env, epochs=500, tensorboard_log='./runs/')

agent.learn()
```