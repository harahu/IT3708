from env import FlatlandEnv
from agents import *

def main():
    score = 0
    for _ in range(1000):
        env = FlatlandEnv(1)
        baseAgent = BaseAgent(env)
        while baseAgent.done == False:
            baseAgent.step()
        score += baseAgent.env.accumulated_reward
    print(score/1000)

def main1():
    nrounds = 100
    agent = SupervisedAgent(FlatlandEnv(1))
    for _ in range(nrounds):
        envs = [FlatlandEnv(1) for i in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        print(meanscore)

def main2():
    nrounds = 100
    agent = ReinforcementAgent(FlatlandEnv(1))
    for _ in range(nrounds):
        envs = [FlatlandEnv(1) for i in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        print(meanscore)

def main3():
    nrounds = 100
    agent = ReinforcementAgent(FlatlandEnv(3))
    for _ in range(nrounds):
        envs = [FlatlandEnv(3) for i in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        print(meanscore)

if __name__ == '__main__':
    main3()