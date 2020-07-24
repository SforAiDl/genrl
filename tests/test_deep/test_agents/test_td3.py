import shutil

from genrl import TD3
from genrl.deep.common import (
    MlpActorCritic,
    OffPolicyTrainer,
    OrnsteinUhlenbeckActionNoise,
)
from genrl.environments import VectorEnv


def test_td3():
    env = VectorEnv("Pendulum-v0", 2)
    algo = TD3("mlp", env, noise=OrnsteinUhlenbeckActionNoise, layers=[1, 1])

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


class custom_net(MlpActorCritic):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_net, self).__init__(
            state_dim, action_dim, kwargs.get("hidden"), val_type="Qsa", discrete=False
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.value = self.critic


def test_td3_custom():
    env = VectorEnv("Pendulum-v0", 2)
    algo = TD3(
        custom_net(
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
