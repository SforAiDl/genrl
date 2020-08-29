import argparse

from genrl.agents import A2C, DDPG, DQN, PPO1, SAC, TD3, VPG
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer, OnPolicyTrainer


def main(args):
    ALGOS = {
        "sac": SAC,
        "a2c": A2C,
        "ppo": PPO1,
        "ddpg": DDPG,
        "td3": TD3,
        "vpg": VPG,
        "dqn": DQN,
    }

    algo = ALGOS[args.algo.lower()]
    env = VectorEnv(
        args.env, n_envs=args.n_envs, parallel=not args.serial, env_type=args.env_type
    )

    logger = get_logger(args.log)
    trainer = None

    if args.algo in ["ppo", "vpg", "a2c"]:
        agent = algo(
            args.arch, env, rollout_size=args.rollout_size
        )  # , batch_size=args.batch_size)
        trainer = OnPolicyTrainer(
            agent,
            env,
            logger,
            epochs=args.epochs,
            render=args.render,
            log_interval=args.log_interval,
        )

    else:
        agent = algo(
            args.arch, env, replay_size=args.replay_size, batch_size=args.batch_size
        )
        trainer = OffPolicyTrainer(
            agent,
            env,
            logger,
            epochs=args.epochs,
            render=args.render,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
        )

    trainer.train()
    trainer.evaluate()


def get_logger(log):
    if "," not in log:
        return [log]
    else:
        log = log.split(",")
        if "" in log or " " in log:
            log = [i for i in log if i != ""]
            log = [i for i in log if i != " "]
        return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep RL algorithms")
    parser.add_argument(
        "-a", "--algo", help="Which Algo to train", default="ppo", type=str
    )
    parser.add_argument(
        "-e", "--env", help="Which env to train on", default="CartPole-v0", type=str
    )
    parser.add_argument(
        "--env-type", help="What kind of env is it", default="gym", type=str
    )
    parser.add_argument(
        "-n",
        "--n-envs",
        help="Number of vectorized envs to train on",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--serial",
        help="Vectorized envs should be serial or parallel",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--epochs", help="How many epochs to train on", default=100, type=int
    )
    parser.add_argument(
        "--render",
        help="Should the env be rendered",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--log", help="Comma separated string of logs", default="stdout", type=str
    )
    parser.add_argument(
        "--arch", help="Which architecture mlp/cnn for now", default="mlp", type=str
    )
    parser.add_argument("--log-interval", help="Set Log interval", default=50, type=int)
    parser.add_argument("--batch-size", help="Batch Size", default=128, type=int)

    offpolicyargs = parser.add_argument_group("Off Policy Args")
    offpolicyargs.add_argument(
        "-ws", "--warmup-steps", help="Warmup steps", default=10000, type=int
    )
    offpolicyargs.add_argument(
        "--replay-size", help="Replay Buffer Size", default=1000, type=int
    )

    onpolicyargs = parser.add_argument_group("On Policy Args")
    onpolicyargs.add_argument(
        "--rollout-size", help="Rollout Buffer Size", default=2048, type=int
    )

    args = parser.parse_args()

    main(args)
