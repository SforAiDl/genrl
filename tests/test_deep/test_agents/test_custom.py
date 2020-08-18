import shutil

from genrl import PPO1, TD3, VPG
from genrl.deep.common import (
    MlpActorCritic,
    MlpPolicy,
    MlpValue,
    NormalActionNoise,
    OffPolicyTrainer,
    OnPolicyTrainer,
    OrnsteinUhlenbeckActionNoise,
)
from genrl.environments import VectorEnv


class custom_policy(MlpPolicy):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_policy, self).__init__(state_dim, action_dim, kwargs.get("hidden"))
        self.state_dim = state_dim
        self.action_dim = action_dim


class custom_actorcritic(MlpActorCritic):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_actorcritic, self).__init__(
            state_dim, action_dim, kwargs.get("hidden")
        )
        self.state_dim = state_dim
        self.action_dim = action_dim


def test_custom_vpg():
    env = VectorEnv("CartPole-v0", 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = custom_policy(state_dim, action_dim, hidden=(64, 64))

    algo = VPG(policy, env)

    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_ppo1_custom():
    env = VectorEnv("CartPole-v0", 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actorcritic = custom_actorcritic(state_dim, action_dim, hidden=(64, 64))

    algo = PPO1(actorcritic, env)

    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
