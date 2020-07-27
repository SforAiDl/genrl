from genrl.bandit import BanditTrainer, CovertypeDataBandit, NeuralGreedyAgent


def test_bandit_trainer():
    bandit = CovertypeDataBandit(download=True)
    agent = NeuralGreedyAgent(bandit)
    trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
    trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
