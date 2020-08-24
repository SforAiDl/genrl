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
    def __init__(self, state_dim, action_dim, policy_layers=(1, 1), **kwargs):
        super(custom_policy, self).__init__(
            state_dim, action_dim, policy_layers=policy_layers, **kwargs
        )


class custom_actorcritic(MlpActorCritic):
    def __init__(
        self,
        state_dim,
        action_dim,
        policy_layers=(1, 1),
        value_layers=(1, 1),
        val_type="V",
        **kwargs
    ):
        super(custom_actorcritic, self).__init__(
            state_dim,
            action_dim,
            policy_layers=policy_layers,
            value_layers=value_layers,
            val_type=val_type,
            **kwargs
        )


def test_custom_vpg():
    env = VectorEnv("CartPole-v0", 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = custom_policy(state_dim, action_dim)

    algo = VPG(policy, env)

    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_custom_ppo1():
    env = VectorEnv("CartPole-v0", 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actorcritic = custom_actorcritic(state_dim, action_dim)

    algo = PPO1(actorcritic, env)

    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
