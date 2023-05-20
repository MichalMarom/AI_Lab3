# ----------- File Form Lab -----------
import time

from matplotlib import pyplot as plt
import Population
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
COOPERATIVE_PSO = 4


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
        self.results_table = np.zeros((5, self.dimensions + 3), float)

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

        elif algorithm_type == COOPERATIVE_PSO:
            self.solve_with_COOPERATIVE_PSO()

    def solve_with_tabu_search(self):
        curr_time = time.time()
        # print("----- Tabu Search -----")
        self.solution, self.score = TabuSearch.tabu_search_ackley(self)
        # print("solution: ", self.solution.coordinates)
        # print("TOTAL SCORE: ", self.score)
        # print("TOTAL time: ", time.time() - curr_time)
        self.results_table[TABU_SEARCH][0] = TABU_SEARCH
        self.results_table[TABU_SEARCH][1] = round(time.time() - curr_time, 2)
        self.results_table[TABU_SEARCH][2] = round(self.score, 2)
        for i, coord in enumerate(self.solution.coordinates):
            self.results_table[TABU_SEARCH][i+3] = round(coord, 2)

        # print(f"the solutions matrix:\n{self.results_table}")

        return

    def solve_with_aco(self):
        curr_time = time.time()
        # print("----- ACO -----")
        self.solution, self.score = aco.aco_algo_ackley(self)
        # print("solution: ", self.solution)
        # print("TOTAL SCORE: ", self.score)
        # print("TOTAL time: ", time.time() - curr_time)
        self.results_table[ACO][0] = ACO
        self.results_table[ACO][1] = round(time.time() - curr_time, 2)
        self.results_table[ACO][2] = round(self.score, 2)
        for i, coord in enumerate(self.solution):
            self.results_table[ACO][i+3] = round(coord, 2)

        # print(f"the solutions matrix:\n{self.results_table}")
        return

    def solve_with_simulated_anealing(self):
        curr_time = time.time()
        # print("----- Simulated Anealing -----")
        simulated_annealing_instance = SimulatedAnnealing.SimulatedAnnealing()
        self.solution, self.score = simulated_annealing_instance.solve_ackley(self)
        # print("solution: ", self.solution)
        # print("TOTAL SCORE: ", self.score)
        # print("TOTAL time: ", time.time() - curr_time)
        self.results_table[SIMULATED_ANNEALING][0] = SIMULATED_ANNEALING
        self.results_table[SIMULATED_ANNEALING][1] = round(time.time() - curr_time, 2)
        self.results_table[SIMULATED_ANNEALING][2] = round(self.score, 2)
        for i, coord in enumerate(self.solution):
            self.results_table[SIMULATED_ANNEALING][i+3] = round(coord, 2)

        # print(f"the solutions matrix:\n{self.results_table}")
        return

    def solve_with_COOPERATIVE_PSO(self):        
        curr_time = time.time()
        # print("----- PSO -----")
        self.solution, self.total_score = CooperativePSO.cooperative_pso_ackley(self)
        # print("solution: ", self.solution)
        # print("TOTAL SCORE: ", self.total_score)        
        # print("TOTAL time: ", time.time() - curr_time)
        self.results_table[COOPERATIVE_PSO][0] = COOPERATIVE_PSO
        self.results_table[COOPERATIVE_PSO][1] = round(time.time() - curr_time, 2)
        self.results_table[COOPERATIVE_PSO][2] = round(self.score, 2)
        for i, coord in enumerate(self.solution):
            self.results_table[COOPERATIVE_PSO][i+3] = round(coord, 2)
        # print(f"the solutions matrix:\n{self.results_table}")
        return

    def solve_with_islands_genetic_algo(self):
        curr_time = time.time()
        # print("----- Islands Genetic Algorithem -----")
        self.solution, self.score = Population.solve_ackley(self)
        # print("solution: ", self.solution)
        # print("TOTAL SCORE: ", self.score)
        # print("TOTAL time: ", time.time() - curr_time)
        self.results_table[ISLANDS][0] = ISLANDS
        self.results_table[ISLANDS][1] = round(time.time() - curr_time, 2)
        self.results_table[ISLANDS][2] = round(self.score, 2)
        for i, coord in enumerate(self.solution):
            self.results_table[ISLANDS][i+3] = round(coord, 2)
        # print(f"the solutions matrix:\n{self.results_table}")
        return

    def print_algorithm_comparison(self):
        column_titles = ['Algorithm', 'Running Time', 'Score'] + [f'D-{i}' for i in range(self.results_table.shape[1] - 3)]
        row_titles = ["TABU_SEARCH", "ACO", "SIMULATED_ANNEALING", "ISLANDS", "COOPERATIVE_PSO"]
        fig, ax = plt.subplots()
        ax.axis('off')

        table = ax.table(cellText = self.results_table,
                        colLabels = column_titles,
                        rowLabels = row_titles,
                        cellLoc = 'center',
                        loc = 'center')

        table.auto_set_font_size(False)
        column_list = []
        [column_list.append(i) for i in range(self.results_table.shape[1])]
        table.auto_set_column_width(column_list)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.show()
        return

