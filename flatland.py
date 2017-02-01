import pygame, sys, math
from pygame.locals import *

from env import FlatlandEnv
from agents import *

# set up the colors
BLACK = (0, 0, 0)
GREY = (130, 130, 130)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CELL_COLOR = {0: WHITE, 1: GREY, 2: GREEN, 3: RED}

#defining cell parameters
CELLNUM = 12
CELLWIDTH = 51
CELLHEIGHT = 51

#deriving window parameters
WINDOWWIDTH = (CELLWIDTH+1)*CELLNUM-1
WINDOWHEIGHT = (CELLHEIGHT+1)*CELLNUM-1

def top_left(y, x):
    return (CELLHEIGHT+1)*x, (CELLWIDTH+1)*y

def center(y, x):
    center_y = (CELLHEIGHT+1)*y + math.ceil(CELLHEIGHT/2)
    center_x = (CELLWIDTH+1)*x + math.ceil(CELLWIDTH/2)
    return (center_x, center_y)

def center_offset(y, x, direction):
    offset = 15
    center_x, center_y = center(y, x)
    if direction == 0:
        return center_x, center_y-15
    elif direction == 1:
        return center_x+15, center_y
    elif direction == 2:
        return center_x, center_y+15
    else:
        return center_x-15, center_y

def render_env(env):
    world = env.get_render_array()
    for y in range(len(world)):
        for x in range(len(world)):
            corner_x, corner_y = top_left(y, x)
            pygame.draw.rect(windowSurface, WHITE, (corner_x, corner_y, CELLWIDTH, CELLHEIGHT), 0)
            cell_center = center(y, x)
            pygame.draw.circle(windowSurface, CELL_COLOR[world[y][x]], cell_center, math.ceil(CELLWIDTH/3), 0)
    agent = env.get_render_agentpos()
    pygame.draw.circle(windowSurface, BLUE, center(agent[0], agent[1]), math.ceil(CELLWIDTH/4), 0)
    pygame.draw.circle(windowSurface, BLUE, center_offset(agent[0], agent[1], env.agentdir), math.ceil(CELLWIDTH/8), 0)

def run_base_agent(rounds):
    score = 0
    for _ in range(rounds):
        env = FlatlandEnv(1)
        baseAgent = BaseAgent(env)
        while baseAgent.done == False:
            baseAgent.step()
        score += env.accumulated_reward
    score /= rounds
    print("Avg score over "+str(rounds)+" rounds: "+str(score))
    retenv = FlatlandEnv(1)
    retagent = BaseAgent(retenv)
    return retenv, retagent

def run_supervised_agent(rounds):
    agent = SupervisedAgent(FlatlandEnv(1))
    for i in range(rounds):
        envs = [FlatlandEnv(1) for j in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        print("Training round "+str(i+1)+" avg score: "+str(meanscore))
    retenv = FlatlandEnv(1)
    agent.set_env(retenv)
    return retenv, agent

def run_reinforcement_agent(rounds, obsrange=1):
    agent = ReinforcementAgent(FlatlandEnv(obsrange))
    for i in range(rounds):
        envs = [FlatlandEnv(obsrange) for j in range(100)]
        meanscore = 0
        for env in envs:
            agent.set_env(env)
            agent.train()
            meanscore += agent.env.accumulated_reward
        meanscore /= len(envs)
        print("Training round "+str(i+1)+" avg score: "+str(meanscore))
    retenv = FlatlandEnv(obsrange)
    agent.set_env(retenv)
    return retenv, agent


def agent_train(mode, rounds):
    if mode == 1:
        return run_base_agent(rounds)
    elif mode == 2:
        return run_supervised_agent(rounds)
    elif mode == 3:
        return run_reinforcement_agent(rounds)
    else:
        return run_reinforcement_agent(rounds, 3)

def main():
    global windowSurface
    mode = int(input("Please choose mode. Base agent (1), supervised agent (2), reinforment agent with wiew range 1 (3) or 3 (4): "))
    if mode == 1:
        rounds = int(input("Number of rounds to average: "))
    else:
        rounds = int(input("Number of training rounds: "))
    fps = int(input("Animation fps: "))

    env, agent = agent_train(mode, rounds)

    # set up pygame
    pygame.init()
    fpsClock = pygame.time.Clock()

    # set up the window
    windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
    pygame.display.set_caption('Flatland')

    # draw black background
    windowSurface.fill(BLACK)

    # run the game loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        render_env(env)
        if agent.done == False:
            agent.step()

        # draw the window onto the screen
        pygame.display.update()
        fpsClock.tick(fps)

if __name__ == '__main__':
    main()
