# ----------- File Form Lab -----------
import SimulatedAnnealing
import TabuSearch
import aco
import CooperativePSO
# ----------- Python Package -----------
import numpy as np
# ----------- Consts Name  -----------
TABU_SEARCH = 0
ACO = 1
SIMULATED_ANNEALING = 2
ISLANDS = 3
Cooperative_PSO = 4


class AckleyFunction:
    def __init__(self):
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi
        self.dimensions = 10
        self.minimum = [0 for i in range(self.dimensions)]
        self.solution = 0
        self.score = 0
        self.bounds = (-32.768, 32.768)

    def function(self, point):
        sum_point = sum([point.coordinates[i] ** 2 for i in range(self.dimensions)])
        sum_cos = sum([np.cos(self.c * point.coordinates[i]) for i in range(self.dimensions)])
        result = -self.a *\
            np.exp(-self.b * np.sqrt((1/self.dimensions) * sum_point)) -\
            np.exp((1/self.dimensions) * sum_cos) + \
            self.a + np.exp(1)

        return result

    def function_coord(self, point):
        sum_point = sum([point[i] ** 2 for i in range(self.dimensions)])
        sum_cos = sum([np.cos(self.c * point[i]) for i in range(self.dimensions)])
        result = -self.a *\
            np.exp(-self.b * np.sqrt((1/self.dimensions) * sum_point)) -\
            np.exp((1/self.dimensions) * sum_cos) + \
            self.a + np.exp(1)

        return result

    def objective_function(self, point):
        dist = np.sqrt(sum([point.coordinates[i]**2 for i in range(self.dimensions)]))
        return dist

    def find_minimum(self, algorithm_type):
        if algorithm_type == TABU_SEARCH:
            self.solve_with_tabu_search()

        elif algorithm_type == ACO:
            self.solve_with_aco()

        elif algorithm_type == SIMULATED_ANNEALING:
            self.solve_with_simulated_anealing()

        elif algorithm_type == ISLANDS:
            self.solve_with_islands_genetic_algo()

        elif algorithm_type == Cooperative_PSO:
            self.solve_with_Cooperative_PSO()

    def solve_with_tabu_search(self):
        print("----- Tabu Search -----")
        self.solution, self.score = TabuSearch.tabu_search_ackley(self)
        print("solution: ", self.solution.coordinates)
        print("TOTAL SCORE: ", self.score)
        return

    def solve_with_aco(self):
        print("----- ACO -----")
        self.solution, self.score = aco.aco_algo_ackley(self)
        print("solution: ", self.solution)
        print("TOTAL SCORE: ", self.score)
        return

    def solve_with_simulated_anealing(self):
        print("----- Simulated Anealing -----")
        # for cluster in self.clusters:
        #     simulated_annealing_instance = SimulatedAnnealing.SimulatedAnnealing(cluster,
        #                                                                          self.start_point,
        #                                                                          self.end_point)
        #     simulated_annealing_instance.simulated_annealing()
        #     solution, score = simulated_annealing_instance.get_solution_and_socre()
        #     self.solution.append(solution)
        #     self.total_score += score

        # print("TOTAL SCORE: ", int(self.total_score))

        simulated_annealing_instance = SimulatedAnnealing.SimulatedAnnealing()
        self.solution, self.score = simulated_annealing_instance.solve_ackley(self)
        print("solution: ", self.solution)
        print("TOTAL SCORE: ", self.score)
        return

    def solve_with_Cooperative_PSO(self):
        print("----- PSO -----")
        self.solution, self.total_score = CooperativePSO.cooperative_pso_ackley(self)
        print("solution: ", self.solution)
        print("TOTAL SCORE: ", self.total_score)
        return

    def solve_with_islands_genetic_algo(self):
        self.total_score = 0
        self.solution = []

        # Initialize the cvrp for the current island
        for i, cluster in enumerate(self.clusters):
            self.islands.append(Population.Population(cluster, self.start_point, self.end_point))

        # Create and start threads for each island
        threads = []
        for i, island in enumerate(self.islands):
            thread = threading.Thread(target=self.islands[i].genetic_algorithm,
                                      args=())
            threads.append(thread)

        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        for i, island in enumerate(self.islands):
            self.solution.append(self.islands[i].get_solution())
            self.total_score += island.best_fitness

        # for path in self.solution:
        #     print("----------------")
        #     for point in path:
        #         print(point.index)

        print("----- Islands Genetic -----")
        print("TOTAL SCORE: ", int(self.total_score))

        return



