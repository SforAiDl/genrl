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
        # DQN 
        algo = DQN("mlp", env)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Double DQN with prioritized replay buffer
        algo1 = DQN("mlp", env, double_dqn=True, prioritized_replay=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo1, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Noisy DQN
        algo2 = DQN("mlp", env, noisy_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo2, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Dueling DQN
        algo3 = DQN("mlp", env, dueling_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo3, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Categorical DQN 
        algo4 = DQN("mlp", env, categorical_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo4, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

    def test_dqn_cnn(self):
        env = gym.make("Breakout-v0")
        # DQN 
        algo = DQN("cnn", env)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Double DQN with prioritized replay buffer
        algo1 = DQN("cnn", env, double_dqn=True, prioritized_replay=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo1, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Noisy DQN
        algo2 = DQN("cnn", env, noisy_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo2, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Dueling DQN
        algo3 = DQN("cnn", env, dueling_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo3, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")

        # Categorical DQN 
        algo4 = DQN("cnn", env, categorical_dqn=True)
        logger = Logger("./logs", ["csv"])

        trainer = OffPolicyTrainer(algo4, env, logger, epochs=1, render=False)
        trainer.train()
        shutil.rmtree("./logs")
