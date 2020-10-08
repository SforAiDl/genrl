import numpy as np


class ValueIterator:
    """Value Iteration Algorithm

    Paper - https://www.ics.uci.edu/~dechter/publications/r42a-mdp_report.pdf

    Args:
        env (gym.Env): Environment with which agent interacts.
        epsilon (float, optional): exploration coefficient for epsilon-greedy exploration.
        gamma (float, optional): discount factor.
    """

    def __init__(self, env, epsilon=0.9, gamma=0.95):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.V = np.zeros(self.env.observation_space.n)
        self.dynamics_model = self.env.P

    def get_action(self, state, explore=True):
        """

        Args:
            state (np.ndarray): The state the agent is currently in
            explore (bool, optional): True is agent is allowed to explore, else False

        Returns:
            np.ndarray: action
        """
        if explore:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        Qs = self._lookahead(state)
        best_action = np.argmax(Qs)
        return best_action

    def update(self, transition):
        """One Step of value iteration

        Args:
            transition (Tuple): Containing the state, action, reward, next_state
        """
        for s in range(self.env.observation_space.n):
            Qs = self._lookahead(s)
            best_action = np.argmax(Qs)
            self.V[s] = Qs[best_action]

    def _lookahead(self, state):
        """Performs a lookahead from the state to compute the q values of the state

        Args:
            state (np.ndarray): The state the agent is currently in

        Returns:
            Qs (np.ndarray): Q values for all actions in the current state

        """
        Qs = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for i in range(len(self.dynamics_model[state][a])):
                prob, next_state, reward, _ = self.dynamics_model[state][a][i]
                Qs[a] += prob * (reward + self.gamma * self.V[next_state])
        return Qs
