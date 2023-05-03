# ----------- File Form Lab -----------
import Individual
# ----------- Python Package -----------
import math
import random


# def tabu_search(clusters, start_point):
#
#     max_iterations = 20
#     solution = []
#     best_solution = []
#     for i in range(len(clusters)):
#         solution.append([])
#         best_solution.append([])
#
#     best_score = []
#     for i in range(len(clusters)):
#         best_score.append(float('inf'))
#
#     for iteration in range(max_iterations):
#         for i, cluster in enumerate(clusters):
#             solution[i] = []
#
#             # Chose a random first node
#             first_node = random.sample(cluster.individuals, 1)[0]
#             # Add the first node to the solution path
#             solution[i].append(first_node)
#             # Remove the first node from the list of the optional nodes
#             individuals = cluster.individuals.copy()
#             individuals.remove(first_node)
#             # Add the start node and  end node to the list of the optional nodes
#             individuals.append(start_point)
#
#             circle_path = find_circle_path(solution[i], cluster.individuals, individuals)
#             solution[i] = circle_path
#
#         for i, cluster in enumerate(clusters):
#             score = calc_score(solution[i])
#             if score < best_score[i]:
#                 best_score[i] = score
#                 best_solution[i] = solution[i]
#
#     for i, solution in enumerate(best_solution):
#         solution = fix_circle_to_start(solution)
#         solution.append(start_point)
#         score = calc_score(solution)
#         best_score[i] = score
#         best_solution[i] = solution
#
#     return best_solution, sum(best_score)
#
#
# def calc_score(solution):
#     total_score = 0
#     for j in range(len(solution) - 1):
#         total_score += math.dist(solution[j].coordinates, solution[j + 1].coordinates)
#
#     return total_score
#
#
# def find_circle_path(solution_path, individuals, update_individuals):
#
#     current_node = solution_path[0]
#     while len(solution_path) < len(individuals):
#         diff = len(individuals) - len(solution_path) - 1
#         next_node = Local_search(update_individuals, current_node, solution_path, tabu_list_size=diff, tabu_time=3, max_iterations=10)
#         solution_path.append(next_node)
#         update_individuals.remove(next_node)
#         current_node = next_node
#     solution_path.append(update_individuals[0])
#
#     return solution_path
#
#
# def fix_circle_to_start(circle_path):
#     nodes_index = [node.index for node in circle_path]
#     start_index = nodes_index.index(0)
#     path = circle_path[start_index:] + circle_path[:start_index]
#     return path
#
#
# # Define the tabu search function
# def Local_search(individuals, current_node, path_solution, tabu_list_size, tabu_time, max_iterations):
#     # Initialize the tabu list with empty solutions
#     tabu_list = [[current_node, 0]]
#
#     # Initialize the best solution and its objective function value
#     best_solution = None
#     best_value = float('inf')
#
#     # Iterate for a maximum number of iterations
#     for i in range(max_iterations):
#         neighborhood = valid_nodes(individuals, tabu_list)
#         next_node = select_next_node(current_node, neighborhood)
#
#         # If there is no node that can be progressed to because all the nodes in the tabu list
#         if next_node is None:
#             # Remove a node from the tabu list that has been sitting there the longest
#             next_node, tabu_list = oldest_node_in_tabu(tabu_list, current_node)
#
#         solution_next_node_value = objective_function(path_solution, next_node)
#         tabu_list = add_node_tabu_list(tabu_list, tabu_list_size, next_node, i)
#
#         if solution_next_node_value < best_value:
#             best_solution = next_node
#             best_value = solution_next_node_value
#         tabu_list = update_tabu_list(tabu_list, tabu_time, i)
#
#     # Return the best solution that found
#     return best_solution
#
#
# # Returns the oldest node form the tabu list
# def oldest_node_in_tabu(tabu_list, current_node):
#
#     if tabu_list[0][0] == current_node:
#         node = tabu_list[1][0]
#         tabu_list.remove(tabu_list[1])
#     else:
#         node = tabu_list[0][0]
#         tabu_list.remove(tabu_list[0])
#
#     return node, tabu_list
#
#
# # Returns the valid nodes for selection that are not in the tabu list
# def valid_nodes(individuals, tabu_list):
#     valid_nodes_list = []
#     tabu_list_coord = [ind[0].coordinates for ind in tabu_list]
#
#     for i, ind in enumerate(individuals):
#         if ind.coordinates not in tabu_list_coord:
#             valid_nodes_list.append(individuals[i])
#
#     return valid_nodes_list
#
#
# # Selects the next closest node to the current node
# def select_next_node(current_node, neighborhood):
#     dist = [math.dist(current_node.coordinates, ind.coordinates) for ind in neighborhood]
#     if dist:
#         next_node_index = dist.index(min(dist))
#         return neighborhood[next_node_index]
#     else:
#         return None
#
#
# # Adding a node to the tabu list and updating the nodes in it
# def add_node_tabu_list(tabu_list, tabu_list_size, new_node, current_time):
#     if len(tabu_list) == tabu_list_size:
#         tabu_list.remove(tabu_list[0])
#     tabu_list.append([new_node, current_time])
#     return tabu_list
#
#
# # Define the objective function to be minimized (total distance)
# def objective_function(solution: list, next_node):
#     total_distance = 0
#     for i in range(len(solution) - 1):
#         total_distance += math.dist(solution[i].coordinates, solution[i+1].coordinates)
#     total_distance += math.dist(solution[len(solution) - 1].coordinates, next_node.coordinates)
#     return total_distance


class Edge:
    nodes: list
    length: float

    def __init__(self, nodes):
        self.nodes = nodes
        self.length = math.dist(nodes[0].coordinates, nodes[1].coordinates)


def tabu_search(clusters, start_point):

    max_iterations = 10
    solution = []
    solution_edges = []
    best_solution = []
    flag = False
    for i in range(len(clusters)):
        solution.append([])
        solution_edges.append([])
        best_solution.append([])

    best_score = []
    for i in range(len(clusters)):
        best_score.append(float('inf'))

    while max_iterations > 0 or solution_is_empty(solution):
        max_iterations -= 1
        for i, cluster in enumerate(clusters):
            solution[i] = []
            solution_edges[i] = []

            # Chose a random first node
            first_node = random.sample(cluster.individuals, 1)[0]
            # Add the first node to the solution path
            solution[i].append(first_node)
            # Remove the first node from the list of the optional nodes
            individuals = cluster.individuals.copy()
            individuals.remove(first_node)
            # Add the start node to the list of the optional nodes
            individuals.append(start_point)

            solution[i], solution_edges[i] = find_circle_path(solution[i], solution_edges[i], cluster.individuals, individuals)

        for i, cluster in enumerate(clusters):
            if solution_edges[i]:
                solution[i] = fix_circle_to_start(solution[i])
                last_edge = Edge([solution[i][len(solution[i]) - 1], start_point])
                for edge_path in solution_edges[i]:
                    if lines_intersect(edge_path, last_edge):
                        score = float('inf')
                        flag = True
                        break

                if not flag:
                    solution[i].append(start_point)
                    solution_edges[i].append(last_edge)
                    score = calc_score(solution[i])
            else:
                score = float('inf')

            if score < best_score[i]:
                best_score[i] = score
                best_solution[i] = solution[i]

    for i, solution in enumerate(best_solution):
        # solution = fix_circle_to_start(solution)
        score = calc_score(solution)
        best_score[i] = score
        best_solution[i] = solution

    return best_solution, sum(best_score)


def find_circle_path(solution_path, solution_edges, individuals, update_individuals):

    #while len(solution_path) < len(individuals) - 1:
    while len(update_individuals) > 1:
        diff = len(individuals)+1 - len(solution_path)
        current_node = solution_path[len(solution_path)-1]
        optional_edges = find_optional_edges(current_node, solution_edges, update_individuals)
        if not optional_edges:
            return [], []
        next_edge = Local_search_edges(optional_edges, solution_edges, tabu_list_size=diff, tabu_time=2, max_iterations=10)

        # Update the path solution
        solution_path.append(next_edge.nodes[1])
        solution_edges.append(next_edge)
        update_individuals.remove(next_edge.nodes[1])

    last_edge = Edge([next_edge.nodes[1], update_individuals[0]])
    for edge_path in solution_edges:
        if lines_intersect(edge_path, last_edge):
            return [], []

    solution_path.append(update_individuals[0])
    solution_edges.append(last_edge)

    return solution_path, solution_edges


def find_optional_edges(current_node, solution_edges, individuals):
    optional_edges = [Edge([current_node, ind]) for ind in individuals]
    edges = optional_edges.copy()
    for edge in edges:
        for edge_path in solution_edges:
            if lines_intersect(edge, edge_path):
                optional_edges.remove(edge)
                break

    return optional_edges


def calc_score(solution):
    total_score = sum([math.dist(solution[i].coordinates, solution[i+1].coordinates) for i in range(len(solution)-1)])
    return total_score


def solution_is_empty(solution):
    for item in solution:
        if not item:
            return True
    return False

def fix_circle_to_start(circle_path):
    nodes_index = [node.index for node in circle_path]
    start_index = nodes_index.index(0)
    path = circle_path[start_index:] + circle_path[:start_index]
    return path


# Define the tabu search function
def Local_search_edges(optional_edges, solution_edges, tabu_list_size, tabu_time, max_iterations):
    # Initialize the tabu list with empty solutions
    tabu_list = []

    # Initialize the best solution and its objective function value
    best_solution = None
    best_value = float('inf')

    # Iterate for a maximum number of iterations
    for i in range(max_iterations):
        neighborhood = valid_edges(optional_edges, tabu_list)
        next_edge = select_next_edge(neighborhood)
        # If there is no node that can be progressed to because all the nodes in the tabu list
        if next_edge is None:
            # Remove a node from the tabu list that has been sitting there the longest
            next_edge, tabu_list = oldest_node_in_tabu(tabu_list)

        solution_next_edge_value = objective_function_edge(solution_edges, next_edge)
        tabu_list = add_edge_tabu_list(tabu_list, tabu_list_size, next_edge, i)

        if solution_next_edge_value < best_value:
            best_solution = next_edge
            best_value = solution_next_edge_value
        tabu_list = update_tabu_list(tabu_list, tabu_time, i)

    # Return the best solution that found
    return best_solution


def valid_edges(edges, tabu_list):
    valid_edge_list = []
    tabu_list_nodes = [item[0].nodes for item in tabu_list]

    for i, edge in enumerate(edges):
        if edge.nodes not in tabu_list_nodes:
            valid_edge_list.append(edge)

    return valid_edge_list


# Selects the next closest node to the current node
def select_next_edge(neighborhood):
    dist = [edge.length for edge in neighborhood]
    if dist:
        next_edge_index = dist.index(min(dist))
        return neighborhood[next_edge_index]
    else:
        return None


# Define the objective function to be minimized (total distance)
def objective_function_edge(solution: list, next_edge):
    total_distance = sum([edge.length for edge in solution])
    total_distance += next_edge.length
    return total_distance


# Adding a node to the tabu list and updating the nodes in it
def add_edge_tabu_list(tabu_list, tabu_list_size, next_edge, current_time):
    if len(tabu_list) == tabu_list_size:
        tabu_list.remove(tabu_list[0])
    tabu_list.append([next_edge, current_time])
    return tabu_list


# Update the tabu list by removing individuals that been in the list more than max tabu time
def update_tabu_list(tabu_list, tabu_time, current_time):
    update_tabu = tabu_list.copy()

    # removing individuals
    for i, ind in enumerate(tabu_list):
        if current_time - ind[1] >= tabu_time:
            update_tabu.remove(tabu_list[i])

    # update the time
    for ind in update_tabu:
        ind[1] += 1

    return update_tabu


def oldest_node_in_tabu(tabu_list):

    node = tabu_list[0][0]
    tabu_list.remove(tabu_list[0])

    return node, tabu_list


def lines_intersect(edge_1, edge_2):
    # Compute the directions of the four line segments
    a1 = edge_1.nodes[0]
    b1 = edge_1.nodes[1]
    a2 = edge_2.nodes[0]
    b2 = edge_2.nodes[1]

    d1 = direction(a1, b1, a2)
    d2 = direction(a1, b1, b2)
    d3 = direction(a2, b2, a1)
    d4 = direction(a2, b2, b1)

    # Check if the two line segments intersect
    if (d1 > 0 > d2 or d1 < 0 < d2)  and (d3 > 0 > d4 or d3 < 0 < d4):
        return True

    return False


def direction(p, q, r):
    return (q.coordinates[1] - p.coordinates[1]) * (r.coordinates[0] - q.coordinates[0]) \
           - (q.coordinates[0] - p.coordinates[0]) * (r.coordinates[1] - q.coordinates[1])