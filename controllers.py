import random, math
import numpy as np

class Controller(object):
    """docstring for Controller"""
    def __init__(self, agent):
        self.agent = agent
    
    def suggest_move():
        pass

class BaseController(Controller):
    """docstring for BaseController
    """
    def suggest_move(self):
        valueList = [0, 0.5, 0]
        for i in range(len(self.agent.observation)):
            obs = self.agent.observation[i][0]
            valueList[i] += self.agent.env.get_reward(obs)
        return valueList.index(max(valueList))

class NeuralController(Controller):
    """docstring for NeuralController
    """
    def __init__(self, agent):
        super(NeuralController, self).__init__(agent)
        self.net_input_size = agent.env.N_MOVE_DIRECTIONS*agent.env.obsrange*agent.env.N_CONTENT
        self.net_input_size = agent.env.N_MOVE_DIRECTIONS
        self.inlayer = [0 for _ in range(self.net_input_size)]
        self.outlayer = [0 for _ in range(self.net_output_size)]
        self.weights = np.array([[random.uniform(0, 0.001) for i in range(self.net_input_size)] for j in range(self.net_output_size)])
        self.eta = 0.01 #aka learning rate

    def suggest_move(self):
        #setting up input layer
        flat_observation = [obs for direction in self.agent.observation for obs in direction]
        self.inlayer = [0 for _ in range(self.net_input_size)]
        for i in range(len(flat_observation)):
            hotindex = i*4 + flat_observation[i]
            self.inlayer[hotindex] = 1
        self.inlayer = np.array(self.inlayer)

        #producing output
        self.outlayer = self.weights @ self.inlayer
        return np.argmax(self.outlayer)

    def delta(self, i):
        pass

    def updateWeights(self):
        deltas = np.array([[self.delta(i)] for i in range(self.net_output_size)])
        invec = np.array([self.inlayer])
        wupdates = (deltas @ invec) * self.eta
        self.weights = np.add(self.weights, wupdates)

class SupervisedController(NeuralController):
    """docstring for SupervisedController
    """
    def __init__(self, agent):
        super(SupervisedController, self).__init__(agent)
        self.trainer = BaseController(agent)

    def correct_choice(self, i):
        if i == self.trainer.suggest_move():
            return 1
        return 0

    def delta(self, i):
        expsum = 0
        for out in self.outlayer:
            expsum += math.exp(out)
        return self.correct_choice(i)-(math.exp(self.outlayer[i])/expsum)

class ReinforcementController(NeuralController):
    """docstring for ReinforcementController"""
    def __init__(self, agent):
        super(ReinforcementController, self).__init__(agent)
        self.prev_inlayer = [0 for _ in range(self.net_input_size)]
        self.prev_outlayer = [0 for _ in range(self.net_output_size)]
        self.gamma = 0.9 #aka temporal discount rate
        self.last_move = None

    def state_shift(self, move):
        self.prev_inlayer = self.inlayer
        self.prev_outlayer = self.outlayer
        self.last_move = move
        self.suggest_move()

    def delta(self, i):
        if i != self.last_move:
            return 0
        return self.agent.reward + self.gamma*(max(self.outlayer)) - self.prevoutlayer[self.last_move]
