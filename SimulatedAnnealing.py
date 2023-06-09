# ----------- Python Package -----------
import math
import random

from matplotlib import pyplot as plt
from Ackley import AckleyFunction

from Clustering import Cluster
from Individual import Individual

INITIAL_TEMPERTURE = 100.0
COOLING_RATE = 0.9

class SimulatedAnnealing:
    temperature: float
    cluster: Cluster
    start_point: Individual
    end_point: Individual
    solution: list
    score: float

    def __init__(self, 
                 cluster: Cluster = None,
                 start_point = None, 
                 end_point = None):
        self.temperature = INITIAL_TEMPERTURE
        self.final_temperature = 1
        self.iterations_per_temp = 100
        self.cooling_rate = COOLING_RATE
        self.cluster = cluster
        self.solution = []
        self.score = 0

        if start_point == None:
            self.start_point = Individual([0,0], 0, 0)
        else:
            self.start_point = start_point
        
        if end_point == None:
            self.end_point = Individual([0,0], 0, -1)
        else:
            self.end_point = end_point
            self.end_point.index = len(self.cluster.individuals)+1

    def euclidean_cost(self, tour: list, points: list, 
                       start_point, end_point):
        total_distance = 0
        # print(f"tour type = {type(tour)}\ntour = {tour}")
        # print(f"points type = {type(points)}\npoints = {points}")
        x1, y1 = start_point.coordinates
        x2, y2 = points[tour[0]].coordinates

        for i in range(len(tour)-1):
            x1, y1 = points[tour[i]].coordinates
            x2, y2 = points[tour[i+1]].coordinates
            total_distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Add the distance from the last point in the tour to the end point
        x_last, y_last = points[tour[-1]].coordinates
        x_end, y_end = end_point.coordinates
        total_distance += math.sqrt((x_end-x_last)**2 + (y_end-y_last)**2)

        # print(f"the cost is {total_distance}")
        return total_distance

    def simulated_annealing(self):
        # self.cluster.print_cluster()
        individuals = self.cluster.individuals
        start_point = self.start_point
        end_point = self.end_point
        individuals.append(start_point)
        individuals.append(end_point)
        # print(f"the individuals are: {individuals}")

        tour = list(range(len(individuals)))
        tour.remove(individuals.index(start_point))
        tour.remove(individuals.index(end_point))
        random.shuffle(tour)
        tour.insert(0, individuals.index(start_point))
        tour.append(individuals.index(end_point))
        current_cost = self.euclidean_cost(tour, individuals, start_point, end_point)

        best_tour = tour[:]
        best_cost = current_cost

        scores = []
        scores.append(best_cost)

        # Repeat until the temperature reaches a minimum value or a stopping criterion is met
        while self.temperature > 0.1:
            # Generate a new candidate tour by making a small random change to the current tour
            i, j = random.sample(range(1, len(individuals)-1), 2)
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            # Evaluate the cost of the new candidate tour
            new_cost = self.euclidean_cost(new_tour, individuals, start_point, end_point)
            
            # Accept the new tour if it's better than the current tour
            if new_cost < current_cost:
                tour = new_tour[:]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_tour = tour[:]
                    best_cost = current_cost
            # Accept the new tour with a certain probability if it's worse than the current tour
            else:
                delta = new_cost - current_cost
                acceptance_prob = math.exp(-delta / self.temperature)
                if random.random() < acceptance_prob:
                    tour = new_tour[:]
                    current_cost = new_cost
            
            # Decrease the temperature according to the cooling schedule
            self.temperature *= self.cooling_rate
            scores.append(best_cost)

        
        # Print the  TSP results
        # print("Best tour:", best_tour)
        # print("Best cost:", round(best_cost, 2))
        
        self.solution = best_tour
        self.score = best_cost
        
        self.set_path_from_indexes()
        self.print_scores_grah(scores)
        # print("solution creatd")
        return 

    def set_path_from_indexes(self):
        new_solution = []
        for index in self.solution:
            new_solution.append(self.cluster.individuals[index])
        
        self.solution = new_solution
        
        # print(f"the full solution is {self.solution}")
        return
    
    def get_solution_and_socre(self):
        return self.solution, self.score
    
    def solve_ackley(self, ackley: AckleyFunction):
        # Initialize the current solution randomly within the search space
        current_solution = [random.uniform(-5, 5) for d in range(ackley.dimensions)]
        current_fitness = ackley.function_coord(current_solution)

        # Initialize the best solution as the current solution
        best_solution = current_solution
        best_fitness = current_fitness

        # Initialize the temperature
        temperature = self.temperature
                
        scores = []
        scores.append(current_fitness)

        # Simulated Annealing main loop
        while temperature > self.final_temperature:
            for _ in range(self.iterations_per_temp):
                # Generate a new neighbor solution
                neighbor_solution = self.generate_neighbor(current_solution, step_size=1.0)

                # Calculate the fitness of the new solution
                neighbor_fitness = ackley.function_coord(neighbor_solution)

                # Determine whether to accept the new solution
                if self.acceptance_probability(current_fitness, neighbor_fitness, temperature) > random.random():
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness

                # Update the best solution if necessary
                if neighbor_fitness < best_fitness:
                    best_solution = neighbor_solution
                    best_fitness = neighbor_fitness

            # Cool down the temperature
            temperature *= self.cooling_rate
            scores.append(current_fitness)

        # self.print_scores_grah(scores)
        return best_solution, best_fitness

        
    def generate_neighbor(self, solution, step_size):
        return [xi + random.uniform(-step_size, step_size) for xi in solution]        
    
    def acceptance_probability(self, current_fitness, new_fitness, temperature):
        if new_fitness < current_fitness:
            return 1.0
        else:
            return math.exp((current_fitness - new_fitness) / temperature)

    def print_scores_grah(self, scores: list):
        max_value_x = len(scores)
        max_value_y = max(scores) + 10
        min_value_x = 0
        min_value_y = 0
        ax = plt.axes()
        plt.suptitle("simulated annealing scores")
        ax.set(xlim=(min_value_x, max_value_x),
               ylim=(min_value_y, max_value_y),
               xlabel='iterations',
               ylabel='score')
        iterations = [index for index in range(len(scores))]
        plt.plot(iterations, scores)        
        plt.show()
        return
