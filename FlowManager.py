# ----------- File For Genetic Algorithm -----------
import CVRP
# ----------- Python Package -----------
import time
import threading
# ----------- Consts Name  ----------
NUM_ISLANDS = 2
TABU_SEARCH = 1
SIMULATED_ANNEALING = 2
ISLANDS = 3

class FlowManager:
    cvrp: CVRP
    results: list

    def __init__(self):
        self.total_time = time.time()
        self.cvrp = CVRP.CVRP()
        self.results = []
        self.islands = []
        return

    def print_pop(self):
        self.cvrp.print_pop()
        self.cvrp.print_clusters()
        return

    def print_graph(self):
        self.cvrp.print_graph()
        return

    def solve_CVRP(self):
        self.cvrp.create_clusters()
        self.cvrp.solve_clustrers_TSP(ISLANDS)
        
        return
