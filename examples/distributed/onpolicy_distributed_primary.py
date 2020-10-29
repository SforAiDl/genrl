from genrl.distributed import (
    Master,
    ExperienceServer,
    ParameterServer,
    ActorNode,
    LearnerNode,
)
from genrl.core.policies import MlpPolicy
from genrl.core.values import MlpValue
from genrl.trainers import DistributedTrainer
import gym
import torch.distributed.rpc as rpc
import torch
from genrl.utils import get_env_properties

N_ACTORS = 1
BUFFER_SIZE = 20
MAX_ENV_STEPS = 500
TRAIN_STEPS = 1000


def get_advantages_returns(rewards, dones, values, gamma=0.99, gae_lambda=1):
    buffer_size = len(rewards)
    advantages = torch.zeros_like(values)
    last_value = values.flatten()
    last_gae_lam = 0
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[step + 1]
            next_value = values[step + 1]
        delta = (
            rewards[step]
            + gamma * next_value * next_non_terminal
            - values[step]
        )
        last_gae_lam = (
            delta
            + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )
        advantages[step] = last_gae_lam
    returns = advantages + values
    return advantages, returns


def unroll_trajs(trajectories):
    size = sum([len(traj) for traj in trajectories])
    obs = torch.zeros(size, *trajectories[0].states[0].shape)
    actions = torch.zeros(size, *trajectories[0].actions[0].shape)
    rewards = torch.zeros(size, *trajectories[0].rewards[0].shape)
    dones = torch.zeros(size, *trajectories[0].dones[0].shape)

    i = 0
    for traj in trajectories:
        for j in range(len(traj)):
            obs[i] = traj.states[j]
            actions[i] = traj.actions[j]
            rewards[i] = traj.rewards[j]
            dones[i] = traj.dones[j]

    return obs, actions, rewards, dones


class A2CActor:
    def __init__(self, env, policy, value, policy_optim, value_optim, grad_norm_limit=0.5):
        self.env = env
        self.policy = policy
        self.value = value
        self.policy_optim = policy_optim
        self.value_optim = value_optim
        self.grad_norm_limit = grad_norm_limit

    def select_action(self, obs: torch.Tensor) -> tuple:
        action_probs = self.policy(obs)
        distribution = torch.distributions.Categorical(probs=action_probs)
        action = distribution.sample()
        return action, {"log_probs": distribution.log_prob(action)}

    def update_params(self, trajectories):
        obs, actions, rewards, dones = unroll_trajs(trajectories)
        values = self.value(obs)
        dist = torch.distributions.Categorical(self.policy(obs))
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        advantages, returns = get_advantages_returns(obs, rewards, dones, values)

        policy_loss = -torch.mean(advantages * log_probs) - torch.mean(entropy)
        value_loss = torch.nn.fucntion.mse_loss(returns, values)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_limit)
        self.policy_optim.step()

        self.value_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.values.parameters(), self.grad_norm_limit)
        self.value_optim.step()

    def get_weights(self):
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict()
        }

    def load_weights(self, weights):
        self.policy.load_state_dict(weights["policy"])
        self.value.load_state_dict(weights["value"])

class Trajectory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.__len = 0

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.__len += 1

    def __len__(self):
        return self.__len

class TrajBuffer():
    def __init__(self, size):
        if size <= 0:
            raise ValueError("Size of buffer must be larger than 0")
        self._size = size
        self._memory = []
        self._full = False

    def is_full(self):
        return self._full
    
    def push(self, traj):
        if not self.is_full():
            self._memory.append(traj)
            if len(self._memory) >= self._size:
                self._full = True
        
    def get(self, batch_size=None):
        if batch_size is None:
            return self._memory


def collect_experience(agent, parameter_server, experience_server, learner):
    current_train_step = -1
    while not learner.is_completed():
        traj = Trajectory()
        while not learner.current_train_step > current_train_step:
            pass
        agent.load_weights(parameter_server.get_weights())
        while not experience_server.is_full():
            obs = agent.env.reset()
            done = False
            for _ in range(MAX_ENV_STEPS):
                action = agent.select_action(obs)
                next_obs, reward, done, _ = agent.env.step(action)
                traj.add(obs, action, reward, done)
                obs = next_obs
                if done:
                    break
            experience_server.push(traj)


class MyTrainer(DistributedTrainer):
    def __init__(self, agent, train_steps, log_interval=200):
        super(MyTrainer, self).__init__(agent)
        self.train_steps = train_steps
        self.log_interval = log_interval
        self._weights_available = True
        self._current_train_step = 0

    @property
    def current_train_step(self):
        return self._current_train_step

    def train(self, parameter_server, experience_server):
        for self._current_train_step in range(self.train_steps):
            if experience_server.is_full():
                self._weights_available = False
                trajectories = experience_server.get()
                if trajectories is None:
                    continue
                self.agent.update_params(1, trajectories)
                parameter_server.store_weights(self.agent.get_weights())
                self._weights_available = True
                if self._current_train_step % self.log_interval == 0:
                    self.evaluate(self._current_train_step)


master = Master(
    world_size=5,
    address="localhost",
    port=29502,
    proc_start_method="fork",
    rpc_backend=rpc.BackendType.TENSORPIPE,
)

env = gym.make("Pendulum-v0")
state_dim, action_dim, discrete, action_lim = get_env_properties(env, "mlp")
policy = MlpPolicy(state_dim, action_dim, (32, 32), discrete)
value = MlpValue(state_dim, action_dim, "V", (32, 32))
policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
value_optim = torch.optim.Adam(value.parameters(), lr=1e-3)
agent = A2CActor(env, policy, value, policy_optim, value_optim)
buffer = TrajBuffer(BUFFER_SIZE)

parameter_server = ParameterServer("param-0", master, agent.get_weights(), rank=1)
experience_server = ExperienceServer("experience-0", master, buffer, rank=2)
trainer = MyTrainer(agent, TRAIN_STEPS)
learner = LearnerNode("learner-0", master, "param-0", "experience-0", trainer, rank=3)
actors = [
    ActorNode(
        name=f"actor-{i}",
        master=master,
        parameter_server_name="param-0",
        experience_server_name="experience-0",
        learner_name="learner-0",
        agent=agent,
        collect_experience=collect_experience,
        rank=i + 4,
    )
    for i in range(N_ACTORS)
]
