import random

class FlatlandEnv():
    """Flatland Environment"""

    ENVSIZE = 10

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
        worldsize = ENVSIZE+2*padsize

        #initiate as world filled with food
        self.world = [[FOOD for i in range(worldsize)] for j in range(worldsize)]

        #build wall padding
        for i in range(padsize):
            self.world[i] = [WALL for i in range(worldsize)]
            self.world[worldsize-i-1] = [WALL for i in range(worldsize)]
        for i in range(worldsize):
            for j in range(padsize):
                self.world[i][j] = WALL
                self.world[i][worldsize-j-1] = WALL

        # fill inn empty and posion cells
        for i in range(padsize, worldsize-padsize):
            for j in range(padsize, worldsize-padsize):
                if random.random() > 0.5:
                    self.world[i][j] = POISON if random.random() < 0.5 else EMPTY

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
        if movdir == 0:
            self.agent_y -= 1
        elif movdir == 1:
            self.agent_x += 1
        elif movdir == 2:
            self.agent_y += 1
        else:
            self.agent_x -= 1

        #updating agent direction
        self.agentdir = movdir

        #updating movecount
        self.moves += 1

        #checking episode end conditions
        if self.moves >= self.maxmoves or get_current_cell_content() == WALL:
            self.done = True

        #log reward
        reward = self.get_reward()
        self.accumulated_reward += reward

        set_current_cell_content(EMPTY)
        observation = self.get_observation()
        return observation, reward, self.done

    def get_observation(self):
        #[[left], [center], [right]]
        north = []
        east = []
        south = []
        west = []
        for i in range(1, obsrange+1):
            north.append(self.world[self.agent_y-i][self.agent_x])
            east.append(self.world[self.agent_y][self.agent_x+i])
            south.append(self.world[self.agent_y+i][self.agent_x])
            west.append(self.world[self.agent_y][self.agent_x-i])
        nh = [north, east, south, west]

        return [nh[(self.agentdir-1)%4], nh[self.agentdir%4], nh[(self.agentdir+1)%4]]

    def get_reward(self):
        reward = FOOD_REWARD
        content = get_current_cell_content()
        if content == EMPTY:
            reward = EMPTY_REWARD
        elif content == POISON:
            reward = POISON_REWARD
        elif content == WALL:
            reward = WALL_REWARD
        return reward