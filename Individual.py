

class Individual:
    coordinates: list
    demand: float
    index: int

    def __init__(self, cord: list, demand: float, index: int):
        self.coordinates = cord
        self.demand = demand
        self.index = int(index)
