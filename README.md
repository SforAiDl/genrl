# genrl
[![pypi](https://img.shields.io/badge/pypi-jigglypuff--rl-blue)](https://pypi.org/project/genrl/)
[![GitHub license](https://img.shields.io/github/license/SforAiDl/genrl)](https://github.com/SforAiDl/genrl/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/SforAiDl/genrl.svg?branch=master)](https://travis-ci.com/SforAiDl/genrl)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/SforAiDl/genrl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SforAiDl/genrl/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/SforAiDl/genrl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SforAiDl/genrl/context:python)
[![codecov](https://codecov.io/gh/SforAiDl/genrl/branch/master/graph/badge.svg)](https://codecov.io/gh/SforAiDl/genrl)

## Installation

We suggest creating a conda virtual environment before installing our package.
```
conda create -n JgRL_env python=3.6 pip
conda activate JgRL_env
```

### From Source (Recommended)
```
git clone https://github.com/SforAiDl/genrl.git
cd genrl
pip install -r requirements.txt
python setup.py install
```

### Using Pip
```
pip install genrl # for most recent stable release
pip install genrl==0.0.1dev2 # for most recent development release
```

## Usage
```python
from genrl import PPO1
import gym

env = gym.make('CartPole-v0')
agent = PPO1(network_type='mlp', env=env, epochs=500, render = True, tensorboard_log='./runs/')

agent.learn()
```
