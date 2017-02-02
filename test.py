import matplotlib.pyplot as plt

from env import FlatlandEnv
from agents import *

def main():
    fig, ax = plt.subplots()
    ep = [i for i in range(1, 51)]
    ax.plot(ep, suptest(50), label='Supervised Agent')
    ax.plot(ep, reitest(50), label='Reinforment Agent')
    ax.plot(ep, reitestex(50), label='Extended Reinforcement Agent')
    ax.plot(ep, [20.5 for i in range(50)], label='Baseline Agent Average')
    legend = ax.legend(loc='lower right', shadow=True)
    plt.xlabel('Round')
    plt.ylabel('Average reward')
    plt.title('Flatland Training Performance')
    ax.grid(True)
    plt.savefig("learning_data.png")
    plt.show()

def suptest(nrounds):
    agent = SupervisedAgent(FlatlandEnv(1))
    scores = []
    for _ in range(nrounds):
        envs = [FlatlandEnv(1) for i in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        scores.append(meanscore)
    return scores

def reitest(nrounds):
    agent = ReinforcementAgent(FlatlandEnv(1))
    scores = []
    for _ in range(nrounds):
        envs = [FlatlandEnv(1) for i in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        scores.append(meanscore)
    print(agent.controller.weights)
    return scores

def reitestex(nrounds):
    agent = ReinforcementAgent(FlatlandEnv(3))
    scores = []
    for _ in range(nrounds):
        envs = [FlatlandEnv(3) for i in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        scores.append(meanscore)
    return scores

if __name__ == '__main__':
    main()
