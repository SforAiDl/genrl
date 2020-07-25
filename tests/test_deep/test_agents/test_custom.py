import shutil

from genrl import PPO1, TD3, VPG
from genrl.deep.common import (
    MlpActorCritic,
    MlpPolicy,
    MlpValue,
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


class custom_multiactorcritic(MlpActorCritic):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_multiactorcritic, self).__init__(
            state_dim, action_dim, kwargs.get("hidden"), val_type="Qsa", discrete=False
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qf2 = MlpValue(
            state_dim, action_dim, val_type="Qsa", hidden=kwargs.get("hidden")
        )
        self.qf1 = self.critic


def test_custom_vpg():
    env = VectorEnv("CartPole-v0", 1)
    algo = VPG(
        custom_policy(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden=(64, 64),
        ),
        env,
    )
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_ppo1_custom():
    env = VectorEnv("CartPole-v0", 1)
    algo = PPO1(
        custom_actorcritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden=(64, 64),
        ),
        env,
    )
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_td3_custom():
    env = VectorEnv("Pendulum-v0", 2)
    algo = TD3(
        custom_multiactorcritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden=(1, 1),
        ),
        env,
        noise=OrnsteinUhlenbeckActionNoise,
    )
    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
