import torch

from genrl.agents import DDPG
from genrl.utils import MultiAgentReplayBuffer, PettingZooInterface, get_model


class MADDPG(ABC):
    """MultiAgent Controller using the MADDPG algorithm

    Attributes:
        network (str): The network type of the Q-value function of the agents.
            Supported types: ["mlp"]
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        shared_layers(:obj:`tuple` of :obj:`int`): Sizes of shared layers in Actor Critic if using
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the critic
        replay_size (int): Capacity of the Replay Buffer
        polyak (float): Target model update parameter (1 for hard update)
        env (Environment): The environment that the agent is supposed to act on
        replay_size (int): Capacity of the Replay Buffer
        render (bool): Should the env be rendered during training?
        noise (:obj:`ActionNoise`): Action Noise function added to aid in exploration
        noise_std (float): Standard deviation of the action noise distribution
        seed (int): Seed for randomness
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        env,
        replay_size: int = int(1e6),
        render: bool = False,
        noise: ActionNoise = None,
        noise_std: float = 0.2,
        warmup_steps=1000,
        **kwargs,
    ):
        self.env = env
        self.network = network
        self.num_agents = self.env.num_agents
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, buffer_maxlen)
        self.EnvInterface = PettingZooInterface(self.env, self.agents)
        self.render = render
        self.warmup_steps = warmup_steps
        self.shared_layers = shared_layers
        ac = self._create_model()
        self.agents = [
            DDPG(ac, noise, noise_std, **kwargs) for agent in self.env.agents
        ]

    def _create_model(self):
        state_dim, action_dim, discrete, _ = self.EnvInterface.get_env_properties()
        if discrete:
            raise Exception(
                "Discrete Environments not supported for {}.".format(__class__.__name__)
            )
        model = get_models("ac", self.network)(
            state_dim, action_dim, self.shared_layers,
        )

    def update(self, batch_size):
        (
            obs_batch,
            indiv_action_batch,
            indiv_reward_batch,
            next_obs_batch,
            global_state_batch,
            global_actions_batch,
            global_next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size)
        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]
            next_global_actions = []
            (
                next_obs_batch_i,
                indiv_next_action,
                next_global_actions,
            ) = self.EnvInterface.trainer(indiv_next_action)
            next_global_actions = torch.cat(
                [next_actions_i for next_actions_i in next_global_actions], 1
            )
            self.EnvInterface.update_agents(
                indiv_reward_batch_i,
                obs_batch_i,
                global_state_batch,
                global_actions_batch,
                global_next_state_batch,
                next_global_actions,
            )

    def train(self, max_episode, max_steps, batch_size):
        episode_rewards = []
        for episode in range(max_episode):
            states = self.env.reset()
            episode_reward = 0
            step = -1
            for step in range(max_steps):
                if self.render:
                    self.env.render(mode="human")

                step += 1
                actions = self.EnvInterface.get_actions(states, steps, warmup_steps)
                next_states, rewards, dones, _ = self.env.step(actions)
                rewards = self.EnvInterface.flatten(rewards)
                episode_reward += np.mean(agent_rewards)
                dones = self.EnvInterface.flatten(dones)
                if all(dones) or step == max_steps - 1:
                    dones = [1 for _ in range(self.num_agents)]
                    self.replay_buffer.push(
                        states, actions, rewards, next_states, dones
                    )
                    episode_rewards.append(episode_reward)
                    print(
                        f"Episode: {episode + 1} | Steps Taken: {step +1} | Reward {episode_reward}"
                    )
                    break
                else:
                    dones = [0 for _ in range(self.num_agents)]
                    self.replay_buffer.push(
                        states, actions, rewards, next_states, dones
                    )
                    states = next_states
                    if len(self.replay_buffer) > batch_size:
                        self.update(batch_size)
