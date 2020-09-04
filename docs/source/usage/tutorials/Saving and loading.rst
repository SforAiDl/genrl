Saving and Loading Weights and Hyperparameters with GenRL
=========================================================

We often want to checkpoint our training model in the RL setting, GenRL offers to save your hyperparameters and weights using TOML and pytorch state_dict respectively. 

Following is a sample code to save checkpoints - 

.. code-block:: python
    import gym
    import shutil
    
    from genrl.agents import VPG
    from genrl.environments.suite import VectorEnv
    from genrl.core import NormalActionNoise
    from genrl.trainers import OnPolicyTrainer

    env = VectorEnv("CartPole-v0", 2)
    algo = VPG("mlp", env, batch_size=5, replay_size=100)

    trainer = OnPolicyTrainer(
        algo,
        env,
        log_mode=["stdout"],
        logdir="./logs",
        save_interval=100,
        epochs=100,
        evaluate_episodes=2,
    )
    trainer.train()
    trainer.evaluate()
    shutil.rmtree("./logs")

Let's say you have a saved weights and hyperparameters file to load onto the model you can change your trainer as below to load it - 

.. code-block:: python
    trainer = OnPolicyTrainer(
        algo,
        env,
        log_mode=["stdout"],
        logdir="./logs",
        save_interval=100,
        epochs=100,
        evaluate_episodes=2,
        load_weights="./checkpoints/VPG_CartPole-v0/1-log-0.pt",
        load_hyperparams="./checkpoints/VPG_CartPole-v0/1-log-0.toml",
    )
    
    