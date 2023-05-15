import math


class Individual:
    coordinates: list
    demand: float
    index: int

    def __init__(self, cord: list, demand: float = 0, index: int = 0):
        self.coordinates = cord
        self.demand = demand
        self.index = int(index)

    def distance_func(self, other_ind):
        return math.dist(self.coordinates, other_ind.coordinates)

    def distance_func_ackley(self, other_ind):
        return math.sqrt(sum([other_ind.coordinates[i]**2 for i in range(len(self.coordinates))]))

