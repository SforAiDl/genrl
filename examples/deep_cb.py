import argparse

from genrl.bandit.run import (
    run_experiment,
    run_multi_algos,
    run_multi_bandits,
    run_single_algos_on_bandit,
)


def main(args):
    if args.algo.lower() == "all" and args.bandit.lower() != "all":
        run_multi_algos(args)
    elif args.algo.lower() != "all" and args.bandit.lower() == "all":
        run_multi_bandits(args)
    elif args.algo.lower() == "all" and args.bandit.lower() == "all":
        raise ValueError("all argument cannot be used for both bandit and algorithm")
    else:
        run_single_algos_on_bandit(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Deep Contextual Bandits")

    parser.add_argument(
        "--run-experiment",
        help="Run pre written experiment with all algos",
        action="store_true",
    )

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
    parser.add_argument("--batch-size", help="Batch Size", default=256, type=int)
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
        "--train-epochs",
        help="Epochs to train for each update",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--log-every",
        help="Timesteps interval for logging",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--logdir",
        help="Directory to store logs in",
        default="./logs/",
        type=str,
    )
    parser.add_argument(
        "--ignore-init",
        help="Initial no. of step to ignore",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--init-train-epochs",
        help="Initial no. of step to ignore",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--train-epochs-decay-steps",
        help="Initial no. of step to ignore",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--download",
        help="Download data for bandit",
        action="store_true",
    )

    args = parser.parse_args()

    if args.run_experiment:
        run_experiment(args)
    else:
        main(args)
