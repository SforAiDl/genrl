import numpy as np

from genrl.agents.modelbased.mcts.mcts import MCTSNode


class UCTNode(MCTSNode):
    def __init__(self, *args, disc_factor, **kwargs):
        super(UCTNode, self).__init__(*args, **kwargs)
        self.disc_factor = disc_factor

    def selection_strategy(self, temp=0):
        if not self.parent:
            return self.get_value()
        return self.get_value() + temperature * self.prior * np.sqrt(
            np.log(self.parent.count) / self.count
        )
