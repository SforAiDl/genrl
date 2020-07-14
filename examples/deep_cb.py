import argparse

import genrl

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
    "bernoulli": genrl.BernoulliCB,
    "gaussian": genrl.GaussianCB,
}


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
    parser.add_argument(
        "--train-epochs", help="Epochs to train for each update", default=20, type=int,
    )
    parser.add_argument(
        "--log-every", help="Timesteps interval for logging", default=1, type=int,
    )

    args = parser.parse_args()

    if args.algo.lower() == "all" and args.bandit.lower() != "all":
        bandit_class = BANDITS[args.bandit.lower()]
        bandit = bandit_class()
        for name, algo in ALGOS.items():
            agent = algo(bandit)
            trainer = genrl.deep.common.trainer.BanditTrainer(agent, bandit)
            trainer.train(
                timesteps=args.timesteps,
                update_interval=args.update_interval,
                update_after=args.update_after,
                batch_size=args.batch_size,
                train_epochs=args.train_epochs,
                log_every=args.log_every,
            )
    elif args.algo.lower() != "all" and args.bandit.lower() == "all":
        for name, bandit_class in BANDITS.items():
            bandit = bandit_class()
            algo = ALGOS[args.algo.lower()]
            agent = algo(bandit)
            trainer = genrl.deep.common.trainer.BanditTrainer(agent, bandit)
            trainer.train(
                timesteps=args.timesteps,
                update_interval=args.update_interval,
                update_after=args.update_after,
                batch_size=args.batch_size,
                train_epochs=args.train_epochs,
                log_every=args.log_every,
            )
    elif args.algo.lower() == "all" and args.bandit.lower() == "all":
        raise ValueError("all argument cannot be used for both bandit and algorithm")

    else:
        algo = ALGOS[args.algo.lower()]
        bandit_class = BANDITS[args.bandit.lower()]
        bandit = bandit_class()
        agent = algo(bandit)
        trainer = genrl.deep.common.trainer.BanditTrainer(agent, bandit)
        trainer.train(
            timesteps=args.timesteps,
            update_interval=args.update_interval,
            update_after=args.update_after,
            batch_size=args.batch_size,
            train_epochs=args.train_epochs,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
