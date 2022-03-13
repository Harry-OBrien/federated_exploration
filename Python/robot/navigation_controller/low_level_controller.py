from .path_planner import Path_Planner

class Path_Not_Found_Error(Exception):
    """Raised when no path can be found to the given target"""
    pass

class Navigation_Controller:
    def __init__(self, start_position, start_orientation, initial_maps):
        self._path_planner = Path_Planner(map_width=len(initial_maps["obstacles"][0]), map_height=len(initial_maps["obstacles"]))
        self._current_goal = None
        self._next_step = 1
        self._current_location = start_position
        self._orientation = start_orientation
        self._maps = initial_maps

    def update_maps(self, new_maps):
        """
        Updates our current local maps with new ones.

        # Arguments
            new_maps: Dict { String:Array<Array<Int>> }
        """
        self._maps = new_maps
        if not self._path_is_legal():
            self._calculate_path()

    def update_location(self, new_location, new_orientation):
        """
        Updates our current location.

        # Arguments
            new_maps: Tuple (Int, Int)
        """
        if self._current_location != new_location:
            self._current_location = new_location
            self._next_step += 1

        self._orientation = new_orientation

        # Check not occupado
        assert (self._maps["obstacles"][new_location] != 1)

    def set_goal(self, new_goal):
        """
        Sets the new goal location in the world.

        # Arguments
            goal_pos: Tuple (Int, Int)
        """
        if self._current_goal != new_goal:
            self._current_goal = new_goal
            try:
                self._calculate_path()
            except Path_Not_Found_Error:
                raise Path_Not_Found_Error()

    def move(self):
        """
        Gets the next action to do based on our current position and the result of the a_star
        search from the path planner.

        # Returns
            Action to do next: Str ("left", "right" "forward")
            or None if no action can be made
        """
        assert self._orientation != None and self._current_location != None

        # If we don't have a path, there is no next action to take
        if self._path == None:
            return None

        # If we're not at our current goal
        if self._current_goal != self._current_location:
            (next_y, next_x) = self._path.path_get(self._next_step)
            
            (y, x) = self._current_location

            delta_x = next_x - x
            delta_y = next_y - y

            assert (delta_x != 0 or delta_y != 0)

            # change in y
            NORTH = 0
            EAST = 1
            SOUTH = 2
            WEST = 3

            if delta_x == 0:
                # Going north (up)
                if delta_y < 0:
                    if self._orientation == NORTH:
                        return "forward"
                    elif self._orientation == EAST:
                        return "left"
                    elif self._orientation == SOUTH or self._orientation == WEST:
                        return "right"
                # Going south (down)
                else:
                    if self._orientation == NORTH or self._orientation == EAST:
                        return "right"
                    elif self._orientation == SOUTH:
                        return "forward"
                    elif self._orientation == WEST:
                        return "left"
            else:
                # Going east (right)
                if delta_x > 0:
                    if self._orientation == NORTH or self._orientation == WEST:
                        return "right"
                    elif self._orientation == EAST:
                        return "forward"
                    elif self._orientation == SOUTH:
                        return "left"
                # going west (left)
                else:
                    if self._orientation == NORTH:
                        return "left"
                    elif self._orientation == EAST or self._orientation == SOUTH:
                        return "right"
                    elif self._orientation == WEST:
                        return "forward"

    def _path_is_legal(self):
        # if any node on our path is occupied, the path is not legal
        if self._path == None:
            return False

        for i in range(self._path.path_len()):
            node = self._path.path_get(i)
            if self._maps["obstacles"][node] == 1:
                return False

        return True

    def _calculate_path(self):
        self._path = self._path_planner.compute_route(self._current_location, self._current_goal, self._maps)
        self._next_step = 1
        if self._path == None:
            raise Path_Not_Found_Error()

        