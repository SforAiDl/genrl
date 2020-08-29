# Vanilla Policy Gradient (VPG)

If you wanted to explore Policy Gradient algorithms in RL, there is a high chance you would've heard of PPO, DDPG, etc. but understanding them can be tricky if you're just starting.

VPG is arguably one of the easiest to understand policy gradient algorithms while still performing to a good enough level.

Let's understand policy gradient at a high level, unlike the classical algorithms like Q-Learning, Monte Carlo where you try to optimise the outputs of the action-value function of the agent which are then used to determine the optimal policy. In policy gradient, as one would like to say we go directly for the kill shot, basically we optimise the thing we want to use at the end, i.e. the Policy.

So that explains the "Policy" part of Policy Gradient, so what about "Gradient", so gradient comes from the fact that we try to optimise the policy by gradient ascent (unlike the popular gradient descent, here we want to increase the values, hence ascent). So that explains the name, but how does it even work.

For that, have a look at the following Psuedo Code (source: [OpenAI](https://spinningup.openai.com))

![Psuedo Code](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)

For a more fundamental understanding [this](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) spinningup article is a good resource

Now that we have an understanding of how VPG works at a high level let's jump into the code to see it in action\
This is a very minimal way to run a VPG agent on **GenRL**

### VPG agent on a Cartpole Environment

```python
import gym  # OpenAI Gym

from genrl import VPG
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("CartPole-v1")
agent = VPG('mlp', env)
trainer = OnPolicyTrainer(agent, env, epochs=200)
trainer.train()
```

This will run a VPG agent `agent` which will interact with the `CartPole-v1` [gym environment](https://gym.openai.com/) \
Let's understand the output on running this (your individual values may differ),

```sh
timestep         Episode          loss             mean_reward
0                0                8.022            19.8835
20480            10               25.969           75.2941
40960            20               29.2478          144.2254
61440            30               25.5711          129.6203
81920            40               19.8718          96.6038
102400           50               19.2585          106.9452
122880           60               17.7781          99.9024
143360           70               23.6839          121.543
163840           80               24.4362          129.2114
184320           90               28.1183          156.3359
204800           100              26.6074          155.1515
225280           110              27.2012          178.8646
245760           120              26.4612          164.498
266240           130              22.8618          148.4058
286720           140              23.465           153.4082
307200           150              21.9764          151.1439
327680           160              22.445           151.1439
348160           170              22.9925          155.7414
368640           180              22.6605          165.1613
389120           190              23.4676          177.316
```

`timestep`: It is basically the units of time the agent has interacted with the environment since the start of training\
`Episode`: It is one complete rollout of the agent, to put it simply it is one complete run until the agent ends up winning or losing\
`loss`: The loss encountered in that episode\
`mean_reward`: The mean reward accumulated in that episode

Now if you look closely the agent will not converge to the max reward even if you increase the epochs to say 5000, it is because that during training the agent is behaving according to a stochastic policy (Meaning when you try to pick from an action given a state from the policy it doesn't simply take the one with the maximum return, rather it samples an action from a probability distribution, so in other words, the policy isn't just like a lookup table, it's function which outputs a probability distribution over the actions which we sample from when using it to pick our optimal action).\
So even if the agent has figured out the optimal policy it is not taking the most optimal action at every step there is an inherent stochasticity to it.\
If we want the agent to make full use of the learnt policy we can add the following line of code at after the training

```python
trainer.evaluate(render=True)
```

This will not only make the agent follow a deterministic policy and thus help you achieve the maximun reward possible reward attainable from the learnt policy but also allow you to see your agent perform by passing `render=True`

For more information on the VPG implementation and the various hyperparameters available have a look at the official **GenRL** docs [here](https://genrl.readthedocs.io/en/latest/api/algorithms/genrl.agents.deep.vpg.html)

Some more implementations

### VPG agent on an Atari Environment

```python
env = VectorEnv("Pong-v0", env_type = "atari")
agent = VPG('cnn', env)
trainer = OnPolicyTrainer(agent, env, epochs=200)
trainer.train()
```
