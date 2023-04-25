# ----------- Python Package -----------
import math
import random
import numpy as np

distance_matrix = np.array([[0, 2, 9, 10],
                            [1, 0, 6, 4],
                            [15, 7, 0, 8],
                            [6, 3, 12, 0]])

def tabu_search_path(clusters):
    solution = []
    for cluster in clusters:
        distance_matrix = init_distance_matrix(cluster)
        solution.append(tabu_search(cluster, tabu_list_size=len(cluster), max_iterations=10, distance_matrix))

def init_distance_matrix(cluster):
    distance_matrix = np.zeros([len(cluster), len(cluster)])
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            distance_matrix[i][j] = math.dist(cluster[i].coordinates, cluster[j].coordinates)

    return distance_matrix

# Define the objective function to be minimized (total distance)
def objective_function(solution: list):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance +=  math.dist(solution[i].coordinates, solution[i+1].coordinates)
    total_distance += math.dist(solution[len(solution) - 1].coordinates, solution[0].coordinates)
    return total_distance

# Define the neighborhood function (swap two cities)
def neighborhood(solution):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

# Define the tabu search function
def tabu_search(cluster, tabu_list_size, max_iterations, distance_matrix):

    # Initialize the tabu list with empty solutions
    tabu_list = [[] for i in range(tabu_list_size)]

    # Initialize the best solution and its objective function value
    best_solution = cluster.individuals
    best_value = objective_function(best_solution, distance_matrix)

    # Initialize the current solution and its objective function value
    current_solution = best_solution
    current_value = best_value

    # Iterate for a maximum number of iterations
    for i in range(max_iterations):
        # Generate a new solution in the neighborhood of the current solution
        new_solution = neighborhood(current_solution)

        # Evaluate the new solution
        new_value = objective_function(new_solution)

        # Check if the new solution is better than the current best solution
        if new_value < best_value:
            best_solution = new_solution
            best_value = new_value

        # Check if the new solution is better than the current solution
        if new_value < current_value:
            current_solution = new_solution
            current_value = new_value
        else:
            # Generate a new solution in the neighborhood of the current solution
            new_solution = neighborhood(current_solution)

            # Evaluate the new solution
            new_value = objective_function(new_solution)

            # Check if the new solution is better than the current solution
            if new_value < current_value:
                current_solution = new_solution
                current_value = new_value
            else:
                # Add the current solution to the tabu list
                tabu_list.pop(0)
                tabu_list.append(current_solution)

                # Generate a new solution in the neighborhood of the current solution
                new_solution = neighborhood(current_solution)
                # Check if the new solution is not in the tabu list
                while new_solution in tabu_list:
                    new_solution = neighborhood(current_solution)

                # Evaluate the new solution
                new_value = objective_function(new_solution)

                # Update the current solution and its objective function value
                current_solution = new_solution
                current_value = new_value

    # Return the best solution found
    return best_solution