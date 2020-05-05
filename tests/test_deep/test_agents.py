import gym, shutil

from genrl import (
    TD3,
    SAC,
    DDPG,
    PPO1,
    VPG,
    DQN,
    OffPolicyTrainer,
    OnPolicyTrainer,
    Logger,
    OrnsteinUhlenbeckActionNoise,
    NormalActionNoise,
)


class TestAlgos:
    def test_sac(self):
        env = gym.make("Pendulum-v0")
        algo = SAC("mlp", env, layers=[1, 1])
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

    def test_td3(self):
        env = gym.make("Pendulum-v0")
        algo = TD3("mlp", env, noise=OrnsteinUhlenbeckActionNoise, layers=[1, 1])
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

    def test_ppo1(self):
        env = gym.make("Pendulum-v0")
        algo = PPO1("mlp", env, layers=[1, 1])
        logger = Logger("./logs", ["csv"])

        trainer = OnPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

    def test_vpg(self):
        env = gym.make("Pendulum-v0")
        algo = VPG("mlp", env, layers=[1, 1])
        logger = Logger("./logs", ["csv"])

        trainer = OnPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

    def test_ddpg(self):
        env = gym.make("Pendulum-v0")
        algo = DDPG("mlp", env, noise=NormalActionNoise, layers=[1, 1])
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

    def test_dqn(self):
        env = gym.make("CartPole-v0")
        algo = DQN("mlp", env, double_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")