# ----------- Python Package -----------
import math
import random

from Clustering import Cluster

# class SimulatedAnealing:
#     temperature: float


# Define the cost function
def cost(tour, individuals):
    total_distance = 0
    for i in range(len(tour)-1):
        x1, y1 = individuals[tour[i]].coordinates
        x2, y2 = individuals[tour[i+1]].coordinates
        total_distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return total_distance

# Define the simulated annealing algorithm
def simulated_annealing(cluster: Cluster, initial_temp, cooling_rate):
    # Initialize the temperature and the initial tour
    temp = initial_temp
    tour = list(range(len(cluster.individuals)))
    random.shuffle(tour)
    current_cost = cost(tour, cluster.individuals)
    best_tour = tour[:]
    best_cost = current_cost
    
    # Repeat until the temperature reaches a minimum value or a stopping criterion is met
    while temp > 0.1:
        # Generate a new candidate tour by making a small random change to the current tour
        i, j = random.sample(range(len(cluster.individuals)), 2)
        new_tour = tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        # Evaluate the cost of the new candidate tour
        new_cost = cost(new_tour, cluster.individuals)
        
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
            acceptance_prob = math.exp(-delta/temp)
            if random.random() < acceptance_prob:
                tour = new_tour[:]
                current_cost = new_cost
        
        # Decrease the temperature according to the cooling schedule
        temp *= cooling_rate
    
    return best_tour, best_cost

    