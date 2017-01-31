import random

class FlatlandEnv(object):
    """Flatland Environment"""

    ENVSIZE = 10
    N_CONTENT = 4
    N_MOVE_DIRECTIONS = 3

    #cell content
    EMPTY = 0
    WALL = 1
    FOOD = 2
    POISON = 3

    #cell rewards
    EMPTY_REWARD = 0
    WALL_REWARD = -100
    FOOD_REWARD = 1
    POISON_REWARD = -4

    #directions
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def __init__(self, obsrange):
        """build a 10x10 flatland with wall padding

        Args:
            obsrange (int): specifies the range that limits how far an agent can view
        """
        self.obsrange = obsrange
        padsize = obsrange+1
        worldsize = self.ENVSIZE+2*padsize

        #initiate as world filled with food
        self.world = [[self.FOOD for i in range(worldsize)] for j in range(worldsize)]

        #build wall padding
        for i in range(padsize):
            self.world[i] = [self.WALL for i in range(worldsize)]
            self.world[worldsize-i-1] = [self.WALL for i in range(worldsize)]
        for i in range(worldsize):
            for j in range(padsize):
                self.world[i][j] = self.WALL
                self.world[i][worldsize-j-1] = self.WALL

        # fill inn empty and posion cells
        for i in range(padsize, worldsize-padsize):
            for j in range(padsize, worldsize-padsize):
                if random.random() > 0.5:
                    self.world[i][j] = self.POISON if random.random() < 0.5 else self.EMPTY

        #place agent somewhere within the padding
        self.agent_x = random.randrange(padsize, worldsize-padsize)
        self.agent_y = random.randrange(padsize, worldsize-padsize)

        #facing a random direction
        self.agentdir = random.randrange(0, 4)

        self.accumulated_reward = 0
        self.moves = 0
        self.maxmoves = 50
        self.done = False

    def get_current_cell_content(self):
        return self.world[self.agent_y][self.agent_x]

    def set_current_cell_content(self, new_content):
        self.world[self.agent_y][self.agent_x] = new_content

    def step(self, action):
        """Move agent one action step in the environment.

        Args:
            action (int):  0 = LEFT, 1 = STRAIGH 2 = RIGHT

        Returns:
            observation (list): agent's observation of the current environment
            reward (int) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        """

        #determining move direction that action corresponds to
        movdir = (self.agentdir + action - 1) % 4

        #determining change in coordinates
        if movdir == self.NORTH:
            self.agent_y -= 1
        elif movdir == self.EAST:
            self.agent_x += 1
        elif movdir == self.SOUTH:
            self.agent_y += 1
        else: #west
            self.agent_x -= 1

        #updating agent direction
        self.agentdir = movdir

        #updating movecount
        self.moves += 1

        #checking episode end conditions
        if self.moves >= self.maxmoves or self.get_current_cell_content() == self.WALL:
            self.done = True

        #log reward
        reward = self.get_reward()
        self.accumulated_reward += reward

        self.set_current_cell_content(self.EMPTY)
        observation = self.get_observation()
        return observation, reward, self.done

    def get_observation(self):
        #[[left], [center], [right]]
        north = []
        east = []
        south = []
        west = []
        for i in range(1, self.obsrange+1):
            north.append(self.world[self.agent_y-i][self.agent_x])
            east.append(self.world[self.agent_y][self.agent_x+i])
            south.append(self.world[self.agent_y+i][self.agent_x])
            west.append(self.world[self.agent_y][self.agent_x-i])
        nh = [north, east, south, west]

        return [nh[(self.agentdir-1)%4], nh[self.agentdir%4], nh[(self.agentdir+1)%4]]

    def get_reward(self, content=None):
        if content==None:
            content = self.get_current_cell_content()
        reward = self.FOOD_REWARD
        if content == self.EMPTY:
            reward = self.EMPTY_REWARD
        elif content == self.POISON:
            reward = self.POISON_REWARD
        elif content == self.WALL:
            reward = self.WALL_REWARD
        return reward

    def get_render_array(self):
        cut = self.obsrange
        return [row[cut:len(self.world)-cut] for row in self.world[cut:len(self.world)-cut]]

    def get_render_agentpos(self):
        cut = self.obsrange
        return [self.agent_y-cut, self.agent_x-cut]
