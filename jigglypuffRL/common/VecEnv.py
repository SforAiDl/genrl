import math
import gym
import numpy as np
import multiprocessing as mp

def worker(parent_conn, child_conn, env):
    parent_conn.close()
    while True:
        cmd, data = child_conn.recv()
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            child_conn.send((observation, reward, done, info))
        elif cmd == 'seed':
            child_conn.send(env.seed(data))
        elif cmd == 'reset':
            observation = env.reset()
            child_conn.send(observation)
        elif cmd == 'render':
            child_conn.send(env.render())
        elif cmd == 'close':
            env.close()
            child_conn.close()
            break
        elif cmd == 'get_spaces':
            child_conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError
          
class VectoredEnv:
    '''
    :param env_fns: (list of Gym environments) List of environments you want to run in parallel  
    '''
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.n_envs = len(env_fns)

        self.procs = []
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for i in range(self.n_envs)])

        for parent_conn, child_conn, env_fn in zip(self.parent_conns, self.child_conns, self.env_fns):
            args = (parent_conn, child_conn, env_fn)
            process = mp.Process(target=worker, args=args, daemon=True)
            process.start()
            self.procs.append(process)
            child_conn.close()

    def get_n_envs(self):
        return self.n_envs

    def get_spaces(self):
        self.parent_conns[0].send(('get_spaces', None))
        observation_space, action_space = self.parent_conns[0].recv()
        return (observation_space, action_space)

    def seed(self, seed=None):
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(('seed', seed + idx))

        return [parent_conn.recv() for parent_conn in self.parent_conns]

    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(('reset',None))

        obs = [parent_conn.recv() for parent_conn in self.parent_conns]
        return obs


    def step(self, actions):
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(('step', action))
        self.waiting = True
        
        result = []
        for parent_conn in self.parent_conns:
            result.append(parent_conn.recv())
        self.waiting = False

        observations, rewards, dones, infos = zip(*result)
        print(observations, rewards, dones)
        return observations, rewards, dones, infos
        
    def close(self):
        if self.waiting:
            for parent_conn in self.parent_conns:
                parent_conn.recv()
        for parent_conn in self.parent_conns:
            parent_conn.send(('close', None))
        for proc in self.procs:
            proc.join()

    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    v = VectoredEnv([env,env])
    v.seed(143)
    v.reset()
    v.step([0,1])
    v.step([1,0])
    v.close()