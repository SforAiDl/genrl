import shutil

from genrl import A2C
from genrl.deep.common import MlpActorCritic, OnPolicyTrainer
from genrl.environments import VectorEnv


def test_a2c():
    env = VectorEnv("CartPole-v0", 1)
    algo = A2C("mlp", env)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_a2c_cnn():
    env = VectorEnv("Pong-v0", 1, env_type="atari")
    algo = A2C("cnn", env)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


class custom_net(MlpActorCritic):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_net, self).__init__(state_dim, action_dim, kwargs.get("hidden"))
        self.state_dim = state_dim
        self.action_dim = action_dim


def test_a2c_custom():
    env = VectorEnv("CartPole-v0", 1)
    algo = A2C(
        custom_net(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden=(64, 64),
        ),
        env,
    )
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
