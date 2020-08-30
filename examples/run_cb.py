from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

import genrl
from genrl.bandit.trainer import DCBTrainer

ALGOS = {
    "bootstrap": genrl.agents.bandits.contextual.BootstrapNeuralAgent,
    "fixed": genrl.agents.bandits.contextual.FixedAgent,
    "linpos": genrl.agents.bandits.contextual.LinearPosteriorAgent,
    "neural-greedy": genrl.agents.bandits.contextual.NeuralGreedyAgent,
    "neural-linpos": genrl.agents.bandits.contextual.NeuralLinearPosteriorAgent,
    "neural-noise": genrl.agents.bandits.contextual.NeuralNoiseSamplingAgent,
    "variational": genrl.agents.bandits.contextual.VariationalAgent,
}
BANDITS = {
    "adult": genrl.utils.data_bandits.AdultDataBandit,
    "census": genrl.utils.data_bandits.CensusDataBandit,
    "covertype": genrl.utils.data_bandits.CovertypeDataBandit,
    "magic": genrl.utils.data_bandits.MagicDataBandit,
    "mushroom": genrl.utils.data_bandits.MushroomDataBandit,
    "statlog": genrl.utils.data_bandits.StatlogDataBandit,
    "bernoulli": genrl.core.bandit.BernoulliMAB,
    "gaussian": genrl.core.bandit.GaussianMAB,
}


def run(args, agent, bandit, plot=True):
    logdir = Path(args.logdir).joinpath(
        f"{agent.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
    )
    trainer = DCBTrainer(
        agent, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )

    results = trainer.train(
        timesteps=args.timesteps,
        update_interval=args.update_interval,
        update_after=args.update_after,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
        init_train_epochs=args.init_train_epochs,
        train_epochs_decay_steps=args.train_epochs_decay_steps,
    )

    if plot:
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle(
            f"{agent.__class__.__name__} on {bandit.__class__.__name__}",
            fontsize=14,
        )
        axs[0, 0].scatter(list(range(len(bandit.regret_hist))), results["regrets"])
        axs[0, 0].set_title("Regret History")
        axs[0, 1].scatter(list(range(len(bandit.reward_hist))), results["rewards"])
        axs[0, 1].set_title("Reward History")
        axs[1, 0].plot(results["cumulative_regrets"])
        axs[1, 0].set_title("Cumulative Regret")
        axs[1, 1].plot(results["cumulative_rewards"])
        axs[1, 1].set_title("Cumulative Reward")
        axs[2, 0].plot(results["regret_moving_avgs"])
        axs[2, 0].set_title("Regret Moving Avg")
        axs[2, 1].plot(results["reward_moving_avgs"])
        axs[2, 1].set_title("Reward Moving Avg")

        fig.savefig(
            Path(logdir).joinpath(
                f"{agent.__class__.__name__}-on-{bandit.__class__.__name__}.png"
            )
        )
        return results


def plot_multi_runs(args, multi_results, title):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
    fig.suptitle(title, fontsize=14)
    axs[0, 0].set_title("Cumulative Regret")
    axs[0, 1].set_title("Cumulative Reward")
    axs[1, 0].set_title("Regret Moving Avg")
    axs[1, 1].set_title("Reward Moving Avg")
    for name, results in multi_results.items():
        axs[0, 0].plot(results["cumulative_regrets"], label=name)
        axs[0, 1].plot(results["cumulative_rewards"], label=name)
        axs[1, 0].plot(results["regret_moving_avgs"], label=name)
        axs[1, 1].plot(results["reward_moving_avgs"], label=name)

    plt.legend()
    fig.savefig(Path(args.logdir).joinpath(f"{title}.png"))


def run_multi_algos(args):
    bandit_class = BANDITS[args.bandit.lower()]
    bandit = bandit_class(download=args.download)
    multi_results = {}
    for name, algo in ALGOS.items():
        agent = algo(bandit)
        multi_results[name] = run(args, agent, bandit)
    plot_multi_runs(args, multi_results, title=f"DCBs-on-{bandit_class.__name__}")


def run_multi_bandits(args):
    algo = ALGOS[args.algo.lower()]
    multi_results = {}
    for name, bandit_class in BANDITS.items():
        bandit = bandit_class(download=args.download)
        agent = algo(bandit)
        multi_results[name] = run(args, agent, bandit)
    plot_multi_runs(args, multi_results, title=f"{algo.__name__}-Performance")


def run_single_algos_on_bandit(args):
    algo = ALGOS[args.algo.lower()]
    bandit_class = BANDITS[args.bandit.lower()]
    bandit = bandit_class(download=args.download)
    agent = algo(bandit)
    run(args, agent, bandit)


def run_experiment(args):
    start_time = datetime.now
    print(f"\nStarting experiment at {start_time():%d-%m-%y %H:%M:%S}\n")
    results = {}

    bandit_class = BANDITS[args.bandit.lower()]
    bandit = bandit_class(download=args.download)

    bootstrap = genrl.BootstrapNeuralAgent(bandit=bandit)
    logdir = Path(args.logdir).joinpath(
        f"{bootstrap.__class__.__name__}-on-{bandit.__class__.__name__}-{start_time():%d%m%y%H%M%S}"
    )
    bootstrap_trainer = DCBTrainer(
        bootstrap, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )
    results["bootstrap"] = bootstrap_trainer.train(
        timesteps=args.timesteps,
        update_interval=50,
        update_after=args.update_after,
        batch_size=args.batch_size,
        train_epochs=100,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
        init_train_epochs=None,
        train_epochs_decay_steps=None,
    )

    fixed = genrl.FixedAgent(bandit)
    logdir = Path(args.logdir).joinpath(
        f"{fixed.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
    )
    fixed_trainer = DCBTrainer(
        fixed, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )
    results["fixed"] = fixed_trainer.train(
        timesteps=args.timesteps,
        update_interval=1,
        update_after=0,
        batch_size=1,
        train_epochs=1,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
        init_train_epochs=None,
        train_epochs_decay_steps=None,
    )

    linpos = genrl.LinearPosteriorAgent(bandit)
    logdir = Path(args.logdir).joinpath(
        f"{linpos.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
    )
    linpos_trainer = DCBTrainer(
        linpos, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )
    results["linpos"] = linpos_trainer.train(
        timesteps=args.timesteps,
        update_interval=1,
        update_after=args.update_after,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
    )

    neural_linpos = genrl.NeuralLinearPosteriorAgent(bandit)
    logdir = Path(args.logdir).joinpath(
        f"{neural_linpos.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
    )
    neural_linpos_trainer = DCBTrainer(
        neural_linpos, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )
    results["neural-linpos"] = neural_linpos_trainer.train(
        timesteps=args.timesteps,
        update_interval=50,
        update_after=args.update_after,
        batch_size=args.batch_size,
        train_epochs=100,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
        init_train_epochs=None,
        train_epochs_decay_steps=None,
    )

    neural_noise = genrl.NeuralNoiseSamplingAgent(bandit)
    logdir = Path(args.logdir).joinpath(
        f"{neural_noise.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
    )
    neural_noise_trainer = DCBTrainer(
        neural_noise, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )
    results["neural-noise"] = neural_noise_trainer.train(
        timesteps=args.timesteps,
        update_interval=50,
        update_after=args.update_after,
        batch_size=args.batch_size,
        train_epochs=100,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
        init_train_epochs=None,
        train_epochs_decay_steps=None,
    )

    variational = genrl.VariationalAgent(bandit)
    logdir = Path(args.logdir).joinpath(
        f"{variational.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
    )
    variational_trainer = DCBTrainer(
        variational, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
    )
    results["variational"] = variational_trainer.train(
        timesteps=args.timesteps,
        update_interval=50,
        update_after=args.update_after,
        batch_size=args.batch_size,
        train_epochs=100,
        log_every=args.log_every,
        ignore_init=args.ignore_init,
        init_train_epochs=10000,
        train_epochs_decay_steps=100,
    )

    plot_multi_runs(args, results, title="Exprimental Results")
    print(f"\nCompleted experiment at {(datetime.now() - start_time).seconds}\n")
