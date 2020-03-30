# JigglypuffRL

## Installation
```
conda env update --name JigglypuffRL --file environment.yml
conda activate JigglypuffRL

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