import shutil

from genrl import (
    TD3,
    SAC,
    DDPG,
    PPO1,
    VPG,
    DQN,
    A2C,
    OffPolicyTrainer,
    OnPolicyTrainer,
)
from genrl.deep.common import (
    OrnsteinUhlenbeckActionNoise,
    NormalActionNoise,
)
from genrl.environments import VectorEnv


class TestAlgos:
    def test_sac(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = SAC("mlp", env, layers=[1, 1])

        trainer = OffPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, render=False
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_td3(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = TD3("mlp", env, noise=OrnsteinUhlenbeckActionNoise, layers=[1, 1])

        trainer = OffPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, evaluate_episodes=2
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

    def test_ppo1(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = PPO1("mlp", env, layers=[1, 1])

        trainer = OnPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, evaluate_episodes=2
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

    def test_vpg(self):
        env = VectorEnv("CartPole-v0", 2)
        algo = VPG("mlp", env, layers=[1, 1])

        trainer = OnPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, evaluate_episodes=2
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

    def test_ddpg(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = DDPG("mlp", env, noise=NormalActionNoise, layers=[1, 1])

        trainer = OffPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, evaluate_episodes=2
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

    def test_dqn(self):
        env = VectorEnv("CartPole-v0", 2)
        # DQN
        algo = DQN("mlp", env)

        trainer = OffPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, evaluate_episodes=2
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

        # # Double DQN with prioritized replay buffer
        # algo1 = DQN("mlp", env, double_dqn=True, prioritized_replay=True)

        # trainer = OffPolicyTrainer(
        #     algo1, log_mode=["csv"], logdir="./logs", epochs=1, render=False
        # )
        # trainer.train()
        # shutil.rmtree("./logs")

        # Noisy DQN
        algo2 = DQN("mlp", env, noisy_dqn=True)

        trainer = OffPolicyTrainer(
            algo2, log_mode=["csv"], logdir="./logs", epochs=1, render=False
        )
        trainer.train()
        shutil.rmtree("./logs")

        # Dueling DQN
        algo3 = DQN("mlp", env, dueling_dqn=True)

        trainer = OffPolicyTrainer(
            algo3, log_mode=["csv"], logdir="./logs", epochs=1, render=False
        )
        trainer.train()
        shutil.rmtree("./logs")

        # Categorical DQN
        algo4 = DQN("mlp", env, categorical_dqn=True)

        trainer = OffPolicyTrainer(
            algo4, log_mode=["csv"], logdir="./logs", epochs=1, render=False
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_a2c(self):
        env = VectorEnv("CartPole-v0", 1)

        # A2C
        algo = A2C("mlp", env)

        trainer = OnPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_dqn_cnn(self):
        env = VectorEnv("Pong-v0", n_envs=2, env_type="atari")

        # DQN
        algo = DQN("cnn", env)

        trainer = OffPolicyTrainer(
            algo, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
        )
        trainer.train()
        shutil.rmtree("./logs")

        # # Double DQN with prioritized replay buffer
        # algo1 = DQN("cnn", env, double_dqn=True, prioritized_replay=True)

        # trainer = OffPolicyTrainer(
        #     algo1, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
        # )
        # trainer.train()
        # shutil.rmtree("./logs")

        # Noisy DQN
        algo2 = DQN("cnn", env, noisy_dqn=True)

        trainer = OffPolicyTrainer(
            algo2, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
        )
        trainer.train()
        shutil.rmtree("./logs")

        # Dueling DQN
        algo3 = DQN("cnn", env, dueling_dqn=True)

        trainer = OffPolicyTrainer(
            algo3, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
        )
        trainer.train()
        shutil.rmtree("./logs")

        # Categorical DQN
        algo4 = DQN("cnn", env, categorical_dqn=True)

        trainer = OffPolicyTrainer(
            algo4, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
        )
        trainer.train()
        shutil.rmtree("./logs")
