from controllers import *

class Agent(object):
    """docstring for BaseAgent
    """
    def __init__(self, env):
        self.env = env
        self.done = env.done
        self.observation = env.get_observation()
        self.reward = 0
        self.controller = None

    def set_env(self, env):
        self.env = env
        self.done = env.done
        self.observation = env.get_observation()
        self.reward = 0

    def step(self, move=None):
        if move == None:
            move = self.controller.suggest_move()
        self.observation, self.reward, self.done = self.env.step(move)

class BaseAgent(Agent):
    """docstring for BaseAgent
    """
    def __init__(self, env):
        super(BaseAgent, self).__init__(env)
        self.controller = BaseController(self)

class SupervisedAgent(Agent):
    """docstring for SupervisedAgent
    """
    def __init__(self, env):
        super(SupervisedAgent, self).__init__(env)
        self.controller = SupervisedController(self)

    def train(self):
        while self.done == False:
            move = self.controller.suggest_move()
            self.controller.update_weights()
            self.step(move)

class ReinforcementAgent(Agent):
    """docstring for ReinforcementAgent
    """
    def __init__(self, env):
        super(ReinforcementAgent, self).__init__(env)
        self.controller = ReinforcementController(self)

    def step(self, move=None):
        if move == None:
            move = self.controller.suggest_move()
        else:
            self.controller.suggest_move()
        self.observation, self.reward, self.done = self.env.step(move)
        self.controller.state_shift(move)

    def train(self):
        while self.done == False:
            self.step()
            self.controller.update_weights()
