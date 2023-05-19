# ----------- File For Genetic Algorithm -----------
import CVRP
import Ackley
# ----------- Python Package -----------
import time
import threading
# ----------- Consts Name  ----------
TABU_SEARCH = 0
ACO = 1
SIMULATED_ANNEALING = 2
ISLANDS = 3
Cooperative_PSO = 4
NUM_ISLANDS = 2
CVRP_problem = 0
ACKLEY_problem = 1


class FlowManager:
    cvrp: CVRP
    results: list

    def __init__(self, ):
        self.total_time = time.time()
        self.cvrp = CVRP.CVRP()
        self.ackley = Ackley.AckleyFunction()
        self.results = []
        self.islands = []
        self.problem = int(input("Enter the problem type:\n CVRP = 0\n Ackley= 1\n"))
        return

    def print_pop(self):
        self.cvrp.print_pop()
        self.cvrp.print_clusters()
        return

    def return_solutions(self):
        if self.problem == CVRP_problem:
            return self.run_multi_thread_CVRP_solution()
        elif self.problem == ACKLEY_problem:
            return self.run_multi_thread_Ackley_solution()

    # ----- CVRP -----
    def run_multi_thread_CVRP_solution(self):
        # Create and start threads for each algorithm
        threads = []
        # Create cluster of nodes for each track
        self.cvrp.create_clusters()
        for i in range(5):
            thread = threading.Thread(target=self.solve_CVRP(i))
            threads.append(thread)
            thread = threading.Thread(target=self.print_graph(i))
            threads.append(thread)

        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return

    def solve_CVRP(self, algorithm_type):
        self.cvrp.solve_clustrers_TSP(algorithm_type)
        return

    def print_graph(self, algorithm_type):
        self.cvrp.print_graph(algorithm_type)
        return

    # ----- Ackley -----
    def run_multi_thread_Ackley_solution(self):
        # Create and start threads for each algorithm
        threads = []
        for i in range(5):
            if i != 3:
                thread = threading.Thread(target=self.find_minimum_ackley(i))
                threads.append(thread)

        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return

    def find_minimum_ackley(self, algorithm_type):
        #ackley_function = Ackley.AckleyFunction()
        self.ackley.find_minimum(algorithm_type)
        return
