# ----------- Python Package -----------
import math
import random

from Clustering import Cluster

INITIAL_TEMPERTURE = 100.0
COOLING_RATE = 0.99

class SimulatedAnnealing:
    temperature: float
    cluster: Cluster
    solution: list
    score: float

    def __init__(self, cluster: Cluster):
        self.temperature = INITIAL_TEMPERTURE
        self.cooling_rate = COOLING_RATE
        self.cluster = cluster
        self.solution = []
        self.score = 0
    
    def euclidean_cost(self, tour: list, points: list):
        total_distance = 0
        # print(f"tour type = {type(tour)}\ntour = {tour}")
        # print(f"points type = {type(points)}\npoints = {points}")

        for i in range(len(tour)-1):
            x1, y1 = points[tour[i]].coordinates
            x2, y2 = points[tour[i+1]].coordinates
            total_distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Add the distance from the last point in the tour to the end point
        x_last, y_last = points[tour[-1]].coordinates
        x_end, y_end = points[-1].coordinates
        total_distance += math.sqrt((x_end-x_last)**2 + (y_end-y_last)**2)

        return total_distance

    def simulated_annealing(self):
        individuals = self.cluster.individuals
        start_point = self.cluster.start_point
        end_point = self.cluster.end_point
        individuals.append(start_point)
        individuals.append(end_point)

        tour = list(range(len(individuals)))
        tour.remove(individuals.index(start_point))
        tour.remove(individuals.index(end_point))
        random.shuffle(tour)
        tour.insert(0, individuals.index(start_point))
        tour.append(individuals.index(end_point))
        current_cost = self.euclidean_cost(tour, individuals)

        best_tour = tour[:]
        best_cost = current_cost
        
        # Repeat until the temperature reaches a minimum value or a stopping criterion is met
        while self.temperature > 0.1:
            # Generate a new candidate tour by making a small random change to the current tour
            i, j = random.sample(range(1, len(individuals)-1), 2)
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            # Evaluate the cost of the new candidate tour
            new_cost = self.euclidean_cost(new_tour, individuals)
            
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
        
        # Print the  TSP results
        print("Best tour:", best_tour)
        print("Best cost:", round(best_cost, 2))
        
        self.solution = best_tour
        self.score = best_cost

        print("solution creatd")
        return 

    def get_solution_and_socre(self):
        return self.solution, self.score