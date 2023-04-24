

class Individual:
    coordinates: list
    demand: float

    def __init__(self, cord: list, demand: float):
        self.coordinates = cord
        self.demand = demand