# ----------- File For Genetic Algorithm -----------
import CVRP
import Ackley
# ----------- Python Package -----------
import time
import threading
# ----------- Consts Name  ----------
NUM_ISLANDS = 2
Tabu_search = 0
ACO = 1
Simulated_Annealing = 2
GA = 3
Cooperative_PSO = 4


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
        self.cvrp.solve_clustrers_TSP(Simulated_Annealing)

        return

    def find_minimum_ackley(self):
        ackley_function = Ackley.AckleyFunction()
        ackley_function.find_minimum(ACO)
