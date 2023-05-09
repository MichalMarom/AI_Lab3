# ----------- File For Genetic Algorithm -----------
import Population
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
    population: Population
    results: list

    def __init__(self):
        self.total_time = time.time()
        self.population = Population.Population()
        self.results = []
        self.islands = []
        return

    def print_pop(self):
        self.population.print_pop()
        self.population.print_clusters()
        return

    def print_graph(self):
        self.population.print_graph()
        return

    def solve_CVRP(self):
        self.population.create_clusters()
        self.population.solve_clustrers_TSP(Cooperative_PSO)

        return
