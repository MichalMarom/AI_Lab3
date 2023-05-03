# ----------- File Form Lab -----------
import Individual
# ----------- Python Package -----------
import math
import random




def tabu_search(clusters, start_point):
    end_point = Individual.Individual(start_point.coordinates, start_point.demand, start_point.index)

    max_iterations = 20
    solution = []
    best_solution = []
    for i in range(len(clusters)):
        solution.append([])
        best_solution.append([])

    best_score = []
    for i in range(len(clusters)):
        best_score.append(float('inf'))

    for iteration in range(max_iterations):
        for i, cluster in enumerate(clusters):
            # # Find the node that closest to the start point
            # dist = [math.dist(start_point.coordinates, ind.coordinates) for ind in cluster.individuals]
            # start_node_index = dist.index(min(dist))
            # solution[i].append(cluster.individuals[start_node_index])
            #
            # individuals = cluster.individuals.copy()
            # individuals.remove(cluster.individuals[start_node_index])
            solution[i] = []
            # Chose a random first node
            first_node = random.sample(cluster.individuals, 1)[0]
            # Add the first node to the solution path
            solution[i].append(first_node)
            # Remove the first node from the list of the optional nodes
            individuals = cluster.individuals.copy()
            individuals.remove(first_node)
            # ADD the start node and  end node to the list of the optional nodes
            individuals.append(start_point)
            # individuals.append(end_point)

            circle_path = find_circle_path(solution[i], cluster.individuals, individuals)
            # solution_path = fix_circle_to_start(circle_path)
            # solution[i] = solution_path
            solution[i] = circle_path

        for i, cluster in enumerate(clusters):
            #solution[i].append(start_point)
            score = calc_score(solution[i])
            if score < best_score[i]:
                best_score[i] = score
                best_solution[i] = solution[i]

    for i, solution in enumerate(best_solution):
        solution = fix_circle_to_start(solution)
        solution.append(start_point)
        score = calc_score(solution)
        best_score[i] = score
        best_solution[i] = solution
            # while len(solution[i]) < len(cluster.individuals):
            #     diff = len(cluster.individuals) - len(solution[i]) - 1
            #     current_node = solution[i][len(solution[i])-1]
            #     next_node = Local_search(individuals, current_node, solution[i], tabu_list_size=diff, tabu_time=2, max_iterations=10)
            #     solution[i].append(next_node)
            #     individuals.remove(next_node)
            # solution[i].append(individuals[0])

    return best_solution, sum(best_score)


def calc_score(solution):
    total_score = 0
    for j in range(len(solution) - 1):
        # Dist from last point of the path to the departure point
        total_score += math.dist(solution[j].coordinates, solution[j + 1].coordinates)


    return total_score


def find_circle_path(solution_path, individuals, update_individuals):

    current_node = solution_path[0]
    while len(solution_path) < len(individuals):
        diff = len(individuals) - len(solution_path) - 1
        next_node = Local_search(update_individuals, current_node, solution_path, tabu_list_size=diff, tabu_time=2, max_iterations=10)
        solution_path.append(next_node)
        update_individuals.remove(next_node)
        current_node = next_node
    solution_path.append(update_individuals[0])

    return solution_path


def fix_circle_to_start(circle_path):
    nodes_index = [node.index for node in circle_path]
    start_index = nodes_index.index(0)
    path = circle_path[start_index:] + circle_path[:start_index]
    return path


def dist_from_start(start_point, individuals):
    dist = [[math.dist(start_point.coordinates, ind.coordinates), i, ind.index] for i, ind in enumerate(individuals)]
    dist = sorted(dist, key=lambda ind: ind[0])
    print(dist)
    return dist[0][1], dist[1][1]


# Define the tabu search function
def Local_search(individuals, current_node, path_solution, tabu_list_size, tabu_time, max_iterations):
    # Initialize the tabu list with empty solutions
    tabu_list = [[current_node, 0]]

    # Initialize the best solution and its objective function value
    best_solution = None
    best_value = float('inf')

    # Initialize the current solution and its objective function value
    # current_solution = best_solution
    # current_value = best_value

    # Iterate for a maximum number of iterations
    for i in range(max_iterations):
        neighborhood = valid_nodes(individuals, tabu_list)
        next_node = select_next_node(current_node, neighborhood)

        # If there is no node that can be progressed to because all the nodes in the tabu list
        if next_node is None:
            # Remove a node from the tabu list that has been sitting there the longest
            next_node, tabu_list = oldest_node_in_tabu(tabu_list, current_node)

        solution_next_node_value = objective_function(path_solution, next_node)
        tabu_list = add_node_tabu_list(tabu_list, tabu_list_size, next_node, i)

        if solution_next_node_value < best_value:
            best_solution = next_node
            best_value = solution_next_node_value
        tabu_list = update_tabu_list(tabu_list, tabu_time, i)
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

    # Return the best solution that found
    return best_solution


def oldest_node_in_tabu(tabu_list, current_node):

    if tabu_list[0][0] == current_node:
        node = tabu_list[1][0]
        tabu_list.remove(tabu_list[1])
    else:
        node = tabu_list[0][0]
        tabu_list.remove(tabu_list[0])

    return node, tabu_list


# Returns the valid nodes for selection that are not in the tabu list
def valid_nodes(individuals, tabu_list):
    valid_nodes_list = []
    tabu_list_coord = [ind[0].coordinates for ind in tabu_list]

    for i, ind in enumerate(individuals):
        if ind.coordinates not in tabu_list_coord:
            valid_nodes_list.append(individuals[i])

    return valid_nodes_list


# Selects the next closest node to the current node
def select_next_node(current_node, neighborhood):
    dist = [math.dist(current_node.coordinates, ind.coordinates) for ind in neighborhood]
    if dist:
        next_node_index = dist.index(min(dist))
        return neighborhood[next_node_index]
    else:
        return None


# Adding a node to the tabu list and updating the nodes in it
def add_node_tabu_list(tabu_list, tabu_list_size, new_node, current_time):
    if len(tabu_list) == tabu_list_size:
        tabu_list.remove(tabu_list[0])
    tabu_list.append([new_node, 0])
    return tabu_list


# Define the objective function to be minimized (total distance)
def objective_function(solution: list, next_node):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += math.dist(solution[i].coordinates, solution[i+1].coordinates)
    total_distance += math.dist(solution[len(solution) - 1].coordinates, next_node.coordinates)
    return total_distance


# Update the tabu list by removing individuals that been in the list more than max tabu time
def update_tabu_list(tabu_list, tabu_time, current_time):
    update_tabu = tabu_list.copy()

    # update the time
    for ind in update_tabu:
        ind[1] += 1

    # removing individuals
    for i, ind in enumerate(tabu_list):
        if current_time - ind[1] >= 0:
            update_tabu.remove(tabu_list[i])
    return update_tabu


# # Define the neighborhood function (swap two nodes)
# def neighborhood(solution):
#     new_solution = solution.copy()
#     i, j = random.sample(range(len(solution)), 2)
#     new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
#     return new_solution


