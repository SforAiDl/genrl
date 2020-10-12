from genrl.agents import NeuralGreedyAgent
from genrl.trainers import DCBTrainer, MABTrainer
from genrl.utils import CovertypeDataBandit

from ..test_agents.test_bandit.utils import write_data


class TestBanditTrainer:
    def test_bandit_trainer(self):
        d = """2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
    2590,56,2,212,-6,390,220,235,151,6225,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
    2804,139,9,268,65,3180,234,238,135,6121,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2
    2785,155,18,242,118,3090,238,238,122,6211,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2
    2595,45,2,153,-1,391,220,234,150,6172,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
    """
        fpath = write_data("covtype.data", d)
        bandit = CovertypeDataBandit(path=fpath)
        agent = NeuralGreedyAgent(bandit)
        trainer = DCBTrainer(agent, bandit, log_mode=["stdout"])
        trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
        fpath.unlink()
