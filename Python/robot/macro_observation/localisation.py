class Invalid_Move_Exception(Exception):
    """Robot cannot move to this tile or in instructed manner"""
    pass

class Localisation:
    """
    The localisation sub-module determines the robot's global position 𝑥𝑖 ∈ 𝑋, with respect to a world coordinate frame.
    """
    def __init__(self, starting_pos, starting_orientation):
        self._pos = starting_pos
        self._orientation = starting_orientation

    def get_state(self):
        return (self._pos, self._orientation)

    def move(self, offset, obstacle_map):
        if offset[0] > 1 or offset[0] < -1\
            or offset[1] > 1 or offset[1] < -1:
            raise Invalid_Move_Exception("Offset too large for agent to travel")

        new_pos = (self._pos[0] + offset[0], self._pos[1] + offset[1])
        if obstacle_map[new_pos] != 1:
            self._pos = new_pos
        else:
            raise Invalid_Move_Exception("New location blocked")

    def change_orientation(self, dir):
        if dir == "left":
            self._orientation = (self._orientation - 1) % 4
        elif dir == "right":
            self._orientation = (self._orientation + 1) % 4
        else:
            raise Invalid_Move_Exception("Invalid rotation direction given")