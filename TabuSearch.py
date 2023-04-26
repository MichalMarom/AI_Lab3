# ----------- Python Package -----------
import math
import random


def tabu_search(clusters, start_point):
    solution = []
    for i in range(len(clusters)):
        solution.append([])

    for i, cluster in enumerate(clusters):
        # Find the node that closest to the start point
        dist = [math.dist(start_point.coordinates, ind.coordinates) for ind in cluster.individuals]
        start_node_index = dist.index(min(dist))
        solution[i].append(cluster.individuals[start_node_index])

        while len(solution[i]) < len(cluster.individuals):
            current_node = solution[len(solution)-1]
            next_node = Local_search(cluster.individuals, current_node, solution[i], tabu_list_size=3, tabu_time=2, max_iterations=10)
            solution[i].append(next_node)
    return solution

# Define the tabu search function
def Local_search(individuals, current_node, path_solution, tabu_list_size, tabu_time, max_iterations):
    # Initialize the tabu list with empty solutions
    tabu_list = [(current_node, 0)]

    # Initialize the best solution and its objective function value
    best_solution = None
    best_value = float('inf')

    # Initialize the current solution and its objective function value
    # current_solution = best_solution
    # current_value = best_value

    # Iterate for a maximum number of iterations
    for i in range(max_iterations):
        legal_nodes = valid_nodes(individuals, tabu_list)
        next_node = select_next_node(current_node, legal_nodes)
        solution_next_node_value = objective_function(path_solution, next_node)
        add_node_tabu_list(tabu_list, tabu_time, tabu_list_size, next_node, i)

        if solution_next_node_value > best_value:
            best_solution = next_node
            best_value = solution_next_node_value

        # # Generate a new solution in the neighborhood of the current solution
        # new_solution = neighborhood(current_solution)
        #
        # # Evaluate the new solution
        # new_value = objective_function(new_solution)
        #
        # # Check if the new solution is better than the current best solution
        # if new_value < best_value:
        #     best_solution = new_solution
        #     best_value = new_value
        #
        # # Check if the new solution is better than the current solution
        # if new_value < current_value:
        #     current_solution = new_solution
        #     current_value = new_value
        # else:
        #     # Generate a new solution in the neighborhood of the current solution
        #     new_solution = neighborhood(current_solution)
        #
        #     # Evaluate the new solution
        #     new_value = objective_function(new_solution)
        #
        #     # Check if the new solution is better than the current solution
        #     if new_value < current_value:
        #         current_solution = new_solution
        #         current_value = new_value
        #     else:
        #         # Add the current solution to the tabu list
        #         tabu_list.append(current_solution)
        #
        #         # Generate a new solution in the neighborhood of the current solution
        #         new_solution = neighborhood(current_solution)
        #         # Check if the new solution is not in the tabu list
        #         while new_solution in tabu_list:
        #             new_solution = neighborhood(current_solution)
        #
        #         # Evaluate the new solution
        #         new_value = objective_function(new_solution)
        #
        #         # Update the current solution and its objective function value
        #         current_solution = new_solution
        #         current_value = new_value

    # Return the best solution found
    return best_solution


# Returns the valid nodes for selection that are not in the tabu list
def valid_nodes(individuals, tabu_list):
    valid_nodes_list = []
    tabu_list_coord = [ind[0].coordinates for ind in tabu_list]
    for i, ind in enumerate(individuals):
        if ind.coordinates not in tabu_list_coord:
            valid_nodes_list.append(individuals[i])
    return valid_nodes_list



# Selects the next closest node to the current node
def select_next_node(current_node, individuals):
    dist = [math.dist(current_node.coordinates, ind.coordinates) for ind in individuals]
    next_node_index = dist.index(min(dist))
    return individuals[next_node_index]


# Adding a node to the tabu list and updating the nodes in it
def add_node_tabu_list(tabu_list, tabu_time, tabu_list_size, new_node, current_iteration):
    if len(tabu_list) == tabu_list_size:
        temp_tabu_list = tabu_list.copy()
        for node in temp_tabu_list:
            if current_iteration - node[1] > tabu_time:
                tabu_list.remove(node)
    tabu_list.append((new_node, current_iteration))
    return tabu_list


# Define the objective function to be minimized (total distance)
def objective_function(solution: list, next_node):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += math.dist(solution[i].coordinates, solution[i+1].coordinates)
    total_distance += math.dist(solution[len(solution) - 1].coordinates, next_node.coordinates)
    return total_distance


# Define the neighborhood function (swap two nodes)
def neighborhood(solution):
    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

