from navigation_controller.low_level_controller import Navigation_Controller
import numpy as np

local_map = np.array([  [0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0]])

class Dumb_Robot:
    def __init__(self, start_pos, start_orientation):
        self._position = start_pos
        self._orientation = start_orientation

        self._maps = {"obstacles":local_map}
        self._navigation = Navigation_Controller(self._position, self._orientation, self._maps)

    def go_to_goal(self, goal):
        self._navigation.set_goal(goal)
        while self._position != goal:
            action = self._navigation.get_next_action()
            self._take_action(action)

    def _take_action(self, action):
        if action == "left":
            self._orientation = (self._orientation - 1) % 4
        elif action == "right":
            self._orientation = (self._orientation + 1) % 4
        elif action == "forward":
            # N, E, S, W
            pos_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            offset = pos_offsets[self._orientation]

            new_pos = (self._position[0] + offset[0], self._position[1] + offset[1])

            # move
            self._position = new_pos
        else:
            raise Exception("unknown action")

        orientations = ["NORTH", "EAST", "SOUTH", "WEST"]
        print("moving to:", self._position, "with orientation:", orientations[self._orientation])
        self._navigation.update_location(self._position, self._orientation)

if __name__ == "__main__":
    robot = Dumb_Robot((7, 7), 1)
    robot.go_to_goal((0, 0))
