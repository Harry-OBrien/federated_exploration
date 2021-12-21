class robot:
    def __init__(self, starting_position, map_size, sensing_range, communication_range):
        self._position = starting_position
        self._map = [[None] * map_size for _ in range(map_size)]
        self._sensing_range = sensing_range
        self._communication_range = communication_range
    
    def move(self, x_offset, y_offset):
        x, y = self._position
        self._position = (x + x_offset, y + y_offset)
        
    def get_map(self):
        return self._map
    
    def union_maps(self, other_map):
        for y, row in enumerate(other_map):
            for x, shared_value in enumerate(row):
                if self._map[y][x] == None:
                    self._map[y][x] = shared_value
                    
    def macro_observation():
        pass
                   