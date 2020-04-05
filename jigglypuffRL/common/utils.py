import gym
import torch


def evaluate(self, num_timesteps=1000):
        s = self.env.reset()
        ep, ep_r, ep_t = 0, 0, 0
        total_r = 0

        print("\nEvaluating...")
        for t in range(num_timesteps):
            a = self.select_action(s)
            s1, r, done, _ = self.env.step(a)
            ep_r += r
            total_r += r
            ep_t += 1

            if done:
                ep += 1
                print("Ep: {}, reward: {}, t: {}".format(ep, ep_r, ep_t))
                s = self.env.reset()
                ep_r, ep_t = 0, 0
            else:
                s = s1

        self.env.close()
        print("Average Reward: {}".format(total_r/num_timesteps))

