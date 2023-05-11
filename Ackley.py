# ----------- File Form Lab -----------
import SimulatedAnnealing
import TabuSearch
import aco
import CooperativePSO
# ----------- Python Package -----------
import math
# ----------- Consts Name  -----------
Tabu_search = 0
ACO = 1
Simulated_Annealing = 2
GA = 3
Cooperative_PSO = 4


class AckleyFunction:
    def __init__(self):
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi
        self.dimensions = 10
        self.minimum = [0 for i in range(self.dimensions)]
        self.solution = 0
        self.score = 0

    def function(self, point):
        sum_point = sum([point.coordinates[i] ** 2 for i in range(self.dimensions)])
        sum_cos = sum([math.cos(self.c * point.coordinates[i]) for i in range(self.dimensions)])
        result = -self.a *\
            math.exp(-self.b * math.sqrt((1/self.dimensions) * sum_point)) -\
            math.exp((1/self.dimensions) * sum_cos) + \
            self.a + math.exp(1)

        return result

    def objective_function(self, point):
        dist = math.sqrt(sum([point.coordinates[i]**2 for i in range(self.dimensions)]))
        return dist

    def find_minimum(self, algorithm_type):
        if algorithm_type == Tabu_search:
            self.solve_with_tabu_search()

        elif algorithm_type == ACO:
            self.solve_with_aco()

        elif algorithm_type == Simulated_Annealing:
            self.solve_with_simulated_anealing()

        #elif algorithm_type == GA:
            # self.solve_with_Cooperative_PSO()

        elif algorithm_type == Cooperative_PSO:
            self.solve_with_Cooperative_PSO()

    def solve_with_tabu_search(self):
        self.solution, self.score = TabuSearch.tabu_search_ackley(self)
        print("TOTAL SCORE: ", self.score)
        return

    def solve_with_aco(self):
        self.solution, self.total_score = aco.aco_algo()
        print("TOTAL SCORE: ", self.total_score)
        return

    def solve_with_simulated_anealing(self):

        for cluster in self.clusters:
            simulated_annealing_instance = SimulatedAnnealing.SimulatedAnnealing(cluster,
                                                                                 self.start_point,
                                                                                 self.end_point)
            simulated_annealing_instance.simulated_annealing()
            solution, score = simulated_annealing_instance.get_solution_and_socre()
            self.solution.append(solution)
            self.total_score += score

        print("TOTAL SCORE: ", int(self.total_score))

        return

    def solve_with_Cooperative_PSO(self):
        self.solution, self.total_score = CooperativePSO.cooperative_pso()
        print("TOTAL SCORE: ", self.total_score)
        return


