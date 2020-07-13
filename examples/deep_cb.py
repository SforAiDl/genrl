import argparse

import genrl
from genrl.deep.common import BanditTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Deep Contextual Bandits")
    parser.add_argument(
        "-a",
        "--algo",
        help="Which algorithm to train",
        default="neural-greedy",
        type=str,
    )
    parser.add_argument(
        "-b", "--bandit", help="Which bandit to train on", default="covertype", type=str
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        help="How many timesteps to train for",
        default=5000,
        type=int,
    )
    parser.add_argument("--batch-size", help="Batch Size", default=128, type=int)
    parser.add_argument(
        "--update-interval", help="Update Interval", default=20, type=int
    )
    parser.add_argument(
        "--update-after",
        help="Timesteps to start updating after",
        default=500,
        type=int,
    )

    args = parser.parse_args()

    ALGOS = {
        "bootstrap": genrl.BootstrapNeuralAgent,
        "fixed": genrl.FixedAgent,
        "linpos": genrl.LinearPosteriorAgent,
        "neural-greedy": genrl.NeuralGreedyAgent,
        "neural-linpos": genrl.NeuralLinearPosteriorAgent,
        "neural-noise": genrl.NeuralNoiseSamplingAgent,
        "variational": genrl.VariationalAgent,
    }
    BANDITS = {
        "adult": genrl.AdultDataBandit,
        "census": genrl.CensusDataBandit,
        "covertype": genrl.CovertypeDataBandit,
        "magic": genrl.MagicDataBandit,
        "mushroom": genrl.MushroomDataBandit,
        "statlog": genrl.StatlogDataBandit,
    }
    bandit_class = BANDITS[args.bandit.lower()]
    algo = ALGOS[args.algo.lower()]
    bandit = bandit_class()
    agent = algo(bandit)
    trainer = genrl.deep.common.trainer.BanditTrainer(agent, bandit)
    trainer.train(
        args.timesteps, args.update_interval, args.update_after, args.batch_size
    )


if __name__ == "__main__":
    main()
