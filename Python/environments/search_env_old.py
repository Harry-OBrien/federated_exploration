import gym
from gym import spaces
import numpy as np
import math

UNKNOWN = 0
EMPTY = 1
FILLED = 2
OTHER_ROBOT = 3
THIS_ROBOT = 4

class Old_Env(gym.Env):
    _random_object_count = 5
    
    def __init__(self, grid_size, vision_distance):
        self._vision_distance = vision_distance
        self._grid_size = grid_size
        self.viewer = None
        self.rendering_grid = [[None] * self._grid_size for _ in range(self._grid_size)]
        
        # Number of actions we can take (up, right, down, left, no_move) (5)
        self.action_space = spaces.Discrete(5)
        
        # Map/grid
        # Can be either this unknown (0), empty(1), filled(2), another robot(3) or us (4).
        self._num_elements = grid_size*grid_size
        self.observation_space = spaces.Box(low=0, high=4, shape=(2, grid_size, grid_size))
        
        # Reset the system
        self.reset()
                
    def step(self, action):
        # Move robot and update state
        self._move_robot(action)
            
        # explore in the square around us
        self._update_observed_space()
                
        # Apply reward (number of explored cells)
        explored_elements = np.count_nonzero(self._explored_grid)
        
        percent_explored = explored_elements / self._num_elements 
        delta_explore = percent_explored - self._last_amount_explored
        self._last_amount_explored = percent_explored
        
        # reward for exploring new ground (delta_explore * 10), penalise (- 0.3) for not 
        reward = delta_explore * 10 - 0.3

        # Check if fully explored
        done = (percent_explored == 1.0)

        # set placeholder for info
        info = {}
        
        return np.array([self._explored_grid, self._robot_grid]), reward, done, info
    
    def _move_robot(self, action):
        # TODO: Robot cannot move into other robot or wall
        x, y = self._robot_position
        # Move up
        if action == 0:
            if (y > 0 and self._master_grid[y-1][x] != FILLED):
                self._robot_position = (x, y - 1)
        
        # Move right
        elif action == 1:
            if(x < self._grid_size - 1 and self._master_grid[y][x + 1] != FILLED):
                self._robot_position = (x + 1, y)
        
        # Move down
        elif action == 2:
            if(y < self._grid_size - 1 and self._master_grid[y + 1][x] != FILLED):
                self._robot_position = (x, y + 1)
        
        # Move left
        elif action == 3:
            if(x > 0  and self._master_grid[y][x - 1] != FILLED):
                self._robot_position = (x - 1, y)
        
        # no move as default/action 4
        else:
            pass

    def _update_observed_space(self):
        x, y = self._robot_position
        x_min = max(x - self._vision_distance, 0)
        x_max = min(x + self._vision_distance, self._grid_size - 1)
        y_min = max(y - self._vision_distance, 0)
        y_max = min(y + self._vision_distance, self._grid_size - 1)
        
        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                self._explored_grid[i][j] = self._master_grid[i][j]
                
        # Update grid of robot positions
        self._robot_grid = [[0] * self._grid_size for _ in range(self._grid_size)]
        self._robot_grid[x][y] = THIS_ROBOT
    
    def render(self, mode='human'):
        screen_size = 550
        square_dimension = 0.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            
            square_dimension = screen_size / self._grid_size
            self.viewer = rendering.Viewer(screen_size, screen_size)
            
            for i, row in enumerate(self._explored_grid):
                for j, value in enumerate(row):
                    l, r, t, b = (
                        j * square_dimension,
                        (j + 1) * square_dimension,
                        i * square_dimension,
                        (i + 1) * square_dimension,
                    )
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    border = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], True)

                    self.rendering_grid[i][j] = square
                    self.viewer.add_geom(square)
                    self.viewer.add_geom(border)
            
        x, y = self._robot_position
        for i, row in enumerate(self._explored_grid):
            for j, value in enumerate(row):
                l, r, t, b = (
                    j * square_dimension,
                    (j + 1) * square_dimension,
                    i * square_dimension,
                    (i + 1) * square_dimension,
                )
                square = self.rendering_grid[i][j]

                if (y == i and x == j):
                    #robot's pos
                    square.set_color(0.8, 0.6, 0.4)
                elif (value == 0):
                    # unkown
                    true_value = self._master_grid[i][j]
                    if (true_value == 1):
                        square.set_color(0.8, 0.8, 0.8)
                    elif (true_value == 2):
                        square.set_color(0.3, 0.3, 0.3)
                    else:
                        square.set_color(0.6, 0.6, 0.6)
                elif (value == 1):
                    # empty
                    square.set_color(1, 1, 1)
                elif (value == 2):
                    #filled
                    square.set_color(0, 0, 0)

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))
    
    def reset(self):
        # Create new arena
        self._create_arena()
        
        # Reset grid and positions
        self._explored_grid = [[UNKNOWN] * self._grid_size for _ in range(self._grid_size)]
        self._last_amount_explored = 0

        # Put robot in random location that is empty (value != FILLED)
        while True:
            x, y = (
                np.random.randint(self._grid_size - 1), 
                np.random.randint(self._grid_size - 1)
            )
            grid_value = self._master_grid[y][x]
            if grid_value != FILLED:
                break
            
        self._robot_position = (x, y)
        
        self._update_observed_space()

        return np.array([self._explored_grid, self._robot_grid])
    
    def _create_arena(self):
        # Currently just 4 random collections in the field
        self._master_grid = [[EMPTY] * self._grid_size for _ in range(self._grid_size)]
        for _ in range(self._random_object_count):
            box_size = np.random.randint(1, 3)
            start_i = np.random.randint(1, self._grid_size - box_size - 1)
            start_j = np.random.randint(1, self._grid_size - box_size - 1)
            
            for y in range(start_i, start_i + box_size + 1):
                for x in range(start_j, start_j + box_size + 1):
                    self._master_grid[y][x] = FILLED
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None