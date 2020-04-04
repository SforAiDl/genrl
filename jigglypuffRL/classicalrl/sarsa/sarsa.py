import gym
import numpy as np

class SARSA:
    '''
    State-Action-Reward-State-Action (SARSA)
    Paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf
    :param env: (Gym environment) The environment to learn from
    :param max_iterations: (int) Maximum number of iterations per episode
    :param max_epsiodes: (int) Maximum number of episodes
    :param epsilon: (float) Epsilon for epsilon-greedy selection of the action
    :param alpha: (float) Learning rate
    :param gamma: (float) Discount Factor
    :param lmbda: (float) Lambda for the eligibility traces
    :param decay_rate_epsilon: (float) Decay rate for the epsilon of epsilon greedy action selection
    :param thres_reward: (float) Threshold of the reward you want the training to stop
    '''
    def __init__(
        self,
        env,
        max_iterations=1000,
        max_episodes=1500001, 
        epsilon=0.9,
        alpha=0.1,
        gamma=0.95,
        lmbda=0.90,
        decay_rate_epsilon=0.999,
        thres_reward=0.8):
        
        self.env = env
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes
        self.epsilon = epsilon
        self.decay_rate_epsilon = decay_rate_epsilon
        self.alpha = alpha 
        self.gamma = gamma
        self.lmbda = lmbda
        self.thres_reward = thres_reward
        
        # Set up the Q table 
        self.Q_table = np.zeros((self.env.observation_space.n,self.env.action_space.n))  
        # Set up eligibility traces
        self.e_table = np.zeros((self.env.observation_space.n,self.env.action_space.n))  
        
    def select_action(self, state):
        # epsilon greedy method to sample actions
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else: 
            action = np.argmax(self.Q_table[state, :])
            
        return action
        
    def update(self, r, state1, action1, state2, action2):
        self.e_table[state1,action1] += 1
        delta = r + self.gamma * self.Q_table[state2, action2] - self.Q_table[state1, action1]
#         self.Q_table[state1,action1] += self.alpha * delta
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                self.Q_table[s,a] = self.Q_table[s,a] + self.alpha * delta * self.e_table[s,a]
                self.e_table[s,a] = self.gamma * self.lmbda * self.e_table[s,a]
        
    def learn(self):
        for ep in range(self.max_episodes):
            t = 0
            
            state1 = self.env.reset()
            action1 = self.select_action(state1)
            
            for t in range(self.max_iterations):
                state2, reward, done, info = self.env.step(action1)
    
                action2 = self.select_action(state2)
                
                self.update(reward, state1, action1, state2, action2)
                
                state1 = state2 
                action1 = action2

                if done:
                    self.epsilon = self.epsilon * self.decay_rate_epsilon
                    break
                
            if ep % 5000 == 0:
                #report every 5000 steps, test 100 games to get avarage point score for statistics and verify if it is solved
                rew_average = 0.
                
                for i in range(100):
                    obs= self.env.reset()
                    done=False
                    while done != True: 
                        action = np.argmax(self.Q_table[obs])
                        obs, rew, done, info = self.env.step(action)
                        rew_average += rew
                rew_average=rew_average/100
                print('Episode {} avarage reward: {}'.format(ep,rew_average))

                if rew_average > self.thres_reward:
                    print(f"{self.env.unwrapped.spec.id} solved")
                    break
        return self.Q_table
        
        
if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    algo = SARSA(env)
    Q_table = algo.learn()
