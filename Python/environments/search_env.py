import gym
from gym import spaces
import numpy as np
import math

class SearchEnv(gym.Env):
    _random_object_count = 48
    _max_timestep = 600
    
    def __init__(self, map_size, vision_distance):
        self._current_timestep = 0

        self._vision_distance = vision_distance
        self._visable_squares = (1 + (self._vision_distance * 2)) ** 2
        self._map_size = map_size
        self._viewer = None
        self._rendering_grid = [[None] * self._map_size for _ in range(self._map_size)]
        
        # Number of actions we can take (up, right, down, left, no_move) (5)
        self.action_space = spaces.Discrete(5)
        
        # Map
        # 4 binary feature maps:
        #   explored space
        #   obstacles
        #   observed robot positions
        #   (goal candidates) not implemented yet
        self._num_elements = self._map_size * self._map_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self._map_size, self._map_size))
        
        # Reset the system
        self.reset()
                
    def step(self, action):
        # Move robot and update state
        self._move_robot(action)
            
        reward = self._calculate_reward()

        # explore in the square around us
        self._update_observed_space()

        # Check if fully explored
        done = self._check_complete()
        if done:
            reward += 100

        # end round if number of timesteps is more than we are allowing
        done = (done or self._current_timestep >= self._max_timestep)
        # print(reward)

        # set placeholder for info
        info = {}

        self._current_timestep += 1
        return np.array([self._exploration_map, self._object_map, self._robot_map]), reward, done, info
    
    def _move_robot(self, action):
        # Do nothing if action = 4 or is out of range
        if action > 3:
            return

        # TODO: Robot cannot move into other robot or wall
        x, y = self._robot_position
        DIRECTION = np.array([
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0)])

        dx, dy = DIRECTION[action]
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x >= self._map_size or\
            new_y < 0 or new_y >= self._map_size:
            return
        
        # If cell is not occupied by object, we can move there
        if self._object_map[new_y][new_x] == 0:
            self._robot_position = (new_x, new_y)

    def _calculate_reward(self):
        unseen_cells = 0
        x, y = self._robot_position
        x_min = max(x - self._vision_distance, 0)
        x_max = min(x + self._vision_distance, self._map_size - 1)
        y_min = max(y - self._vision_distance, 0)
        y_max = min(y + self._vision_distance, self._map_size - 1)
        
        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                # If we haven't seen this cell before
                if self._exploration_map[i][j] == 0:
                    unseen_cells += 1

        return -1 + -self._visable_squares + unseen_cells

    def _check_complete(self):
        explored = np.count_nonzero(self._exploration_map)
        return explored == self._num_elements

    def _update_observed_space(self):
        x, y = self._robot_position
        x_min = max(x - self._vision_distance, 0)
        x_max = min(x + self._vision_distance, self._map_size - 1)
        y_min = max(y - self._vision_distance, 0)
        y_max = min(y + self._vision_distance, self._map_size - 1)
        
        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                self._exploration_map[i][j] = 1
                self._object_map[i][j] = self._master_object_map[i][j]
                
        # Update grid of robot positions
        self._robot_map = [[0] * self._map_size for _ in range(self._map_size)]
        self._robot_map[y][x] = 1
    
    def render(self, mode='human'):
        screen_size = 550
        square_dimension = 0.0

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            
            square_dimension = screen_size / self._map_size
            self._viewer = rendering.Viewer(screen_size, screen_size)
            
            for i in range(self._map_size):
                for j in range(self._map_size):
                    l, r, t, b = (
                        j * square_dimension,
                        (j + 1) * square_dimension,
                        i * square_dimension,
                        (i + 1) * square_dimension,
                    )
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    border = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], True)

                    self._rendering_grid[i][j] = square
                    self._viewer.add_geom(square)
                    self._viewer.add_geom(border)
            
        for i, row in enumerate(self._object_map):
            for j, object_exists in enumerate(row):
                square = self._rendering_grid[i][j]

                # If a robot exists in this square
                if (self._robot_map[i][j] == 1):
                    #robot's pos
                    square.set_color(0.8, 0.6, 0.4)

                # if unexplored
                elif (self._exploration_map[i][j] == 0):
                    true_value = self._master_object_map[i][j]
                    if (true_value == 0):
                        square.set_color(0.8, 0.8, 0.8)
                    else:
                        square.set_color(0.3, 0.3, 0.3)

                # Square is explored and blocked
                elif (object_exists):
                    square.set_color(0, 0, 0)

                # square is explored and empty
                else:
                    square.set_color(1, 1, 1)

        return self._viewer.render(return_rgb_array=(mode == "rgb_array"))
    
    def reset(self):
        self._current_timestep = 0

        # Reset grid and positions
        self._master_object_map = [[0] * self._map_size for _ in range(self._map_size)] # The env's reference for the world
        self._exploration_map   = [[0] * self._map_size for _ in range(self._map_size)] # if a block has been explored
        self._object_map        = [[0] * self._map_size for _ in range(self._map_size)] # if an object exists in the explored space
        self._robot_map         = [[0] * self._map_size for _ in range(self._map_size)] # the last known positions of the robots

        # Create new arena
        self._create_arena()

        # Put robot in random location that is empty (value == 0)
        while True:
            x, y = (
                np.random.randint(self._map_size - 1), 
                np.random.randint(self._map_size - 1)
            )
            occupied = self._master_object_map[y][x]
            # If unoccupied
            if not occupied:
                break
            
        self._robot_position = (x, y)
        self._robot_map[y][x] = 1
        
        self._update_observed_space()

        return np.array([self._exploration_map, self._object_map, self._robot_map])
    
    def _create_arena(self):
        return
        # Binary field where a 1 is an object, 0 is air
        for _ in range(self._random_object_count):
            box_size = np.random.randint(1, 3)
            start_i = np.random.randint(1, self._map_size - box_size - 1)
            start_j = np.random.randint(1, self._map_size - box_size - 1)
            
            for y in range(start_i, start_i + box_size + 1):
                for x in range(start_j, start_j + box_size + 1):
                    self._master_object_map[y][x] = 1
    
    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None