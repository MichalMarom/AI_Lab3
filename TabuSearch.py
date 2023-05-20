# ----------- File Form Lab -----------
import Individual
# ----------- Python Package -----------
import math
import random
import numpy as np
from matplotlib import pyplot as plt


class Edge:
    nodes: list
    length: float

    def __init__(self, nodes):
        self.nodes = nodes
        self.length = math.dist(nodes[0].coordinates, nodes[1].coordinates)


# ----------- Search path for each cluster -----------
def tabu_search(clusters, start_point):
    scores = []
    max_iterations = 100
    solution = []
    solution_edges = []
    best_solution = []
    tabu_time = 1

    for i in range(len(clusters)):
        solution.append([])
        solution_edges.append([])
        best_solution.append([])

    best_score = []
    for i in range(len(clusters)):
        best_score.append(float('inf'))

    while max_iterations > 0:
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

            solution[i], solution_edges[i] = find_path(solution[i], solution_edges[i], cluster.individuals, individuals, tabu_time)
            solution[i], solution_edges[i] = add_last_edge(solution[i], solution_edges[i], start_point)

        for i, cluster in enumerate(clusters):
            score = calc_score(solution[i])
            if score < best_score[i]:
                best_score[i] = score
                best_solution[i] = solution[i]
                tabu_time -= 1
            if score == best_score[i]:
                tabu_time += 1

        if sum(best_score) != float('inf'):
            scores.append(sum(best_score))

    for i, solution in enumerate(best_solution):
        score = calc_score(solution)
        best_score[i] = score
        best_solution[i] = solution

    return best_solution, sum(best_score), scores


# Finding the path and returns the nodes and edges
def find_path(solution_path, solution_edges, individuals, update_individuals, tabu_time):
    while len(update_individuals) > 1:
        diff = len(individuals)+1 - len(solution_path)
        current_node = solution_path[len(solution_path)-1]
        optional_edges = find_optional_edges(current_node, solution_edges, update_individuals)
        if not optional_edges:
            return [], []
        next_edge = Local_search_edges(optional_edges, solution_edges, tabu_list_size=diff, tabu_time=tabu_time, max_iterations=100)

        # Update the path solution
        solution_path.append(next_edge.nodes[1])
        solution_edges.append(next_edge)
        update_individuals.remove(next_edge.nodes[1])

    # Adding the last edge to the path
    last_edge = Edge([next_edge.nodes[1], update_individuals[0]])
    for edge_path in solution_edges:
        if lines_intersect(edge_path, last_edge):
            return [], []

    solution_path.append(update_individuals[0])
    solution_edges.append(last_edge)

    return solution_path, solution_edges


# Returns the optional edges from the current node without intersecting the path edges
def find_optional_edges(current_node, solution_edges, individuals):
    optional_edges = [Edge([current_node, ind]) for ind in individuals]
    edges = optional_edges.copy()
    for edge in edges:
        for edge_path in solution_edges:
            if lines_intersect(edge, edge_path):
                optional_edges.remove(edge)
                break

    return optional_edges


# Adding the last edge between the last node in the path and the first node
def add_last_edge(solution_path, solution_edges, start_point):
    if solution_path:
        solution_path, solution_edges = fix_path_to_start(solution_path, solution_edges)
        # if not solution_path:
        #     return [], []
        last_node = solution_path[len(solution_path) - 1]
        last_edge = Edge([last_node, start_point])
        # for edge_path in solution_edges:
        #     if lines_intersect(edge_path, last_edge):
        #         return [], []
        solution_path.append(start_point)
        solution_edges.append(last_edge)
        return solution_path, solution_edges

    else:
        return [], []


# Correcting the path so that it starts from point 0
def fix_path_to_start(circle_path, solution_edges):
    first_edge = Edge([circle_path[0], circle_path[len(circle_path) - 1]])
    nodes_index = [node.index for node in circle_path]
    start_index = nodes_index.index(0)
    path = circle_path[start_index:] + circle_path[:start_index]
    # for edge_path in solution_edges:
    #     if lines_intersect(edge_path, first_edge):
    #         return [], []
    solution_edges.append(first_edge)
    return path, solution_edges


# Calculate the length of the path
def calc_score(solution):
    if solution:
        total_score = sum([math.dist(solution[i].coordinates, solution[i+1].coordinates) for i in range(len(solution)-1)])
    else:
        total_score = float('inf')

    return total_score


# ----------- Local Search using a tabu list -----------
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


# Returns the legal edges that are not in the tabu list
def valid_edges(edges, tabu_list):
    valid_edge_list = []
    tabu_list_nodes = [item[0].nodes for item in tabu_list]

    for i, edge in enumerate(edges):
        if edge.nodes not in tabu_list_nodes:
            valid_edge_list.append(edge)

    return valid_edge_list


# Selects the shortest edge from the current node
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


# Adding a edge to the tabu list and updating the edges in it
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


# Returning and deleting the oldest node in the taboo list
def oldest_node_in_tabu(tabu_list):

    node = tabu_list[0][0]
    tabu_list.remove(tabu_list[0])

    return node, tabu_list


# ----------- Check for Intersections -----------
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
    if (d1 > 0 > d2 or d1 < 0 < d2) and (d3 > 0 > d4 or d3 < 0 < d4):
        return True

    return False


def direction(p, q, r):
    return (q.coordinates[1] - p.coordinates[1]) * (r.coordinates[0] - q.coordinates[0]) \
           - (q.coordinates[0] - p.coordinates[0]) * (r.coordinates[1] - q.coordinates[1])


# --------------------------------------------------------------------------------------------------------
# ----------- Search Minimum for ackley function -----------

def tabu_search_ackley(ackley):
    max_iterations = 100
    tabu_time = 10
    neighborhood_size = 1000
    tabu_list = []
    tabu_list_size = math.sqrt(max_iterations)
    best_solution = None
    best_score = np.inf

    # Chose a random first node
    first_node_coordinates = [random.uniform(ackley.bounds[0], ackley.bounds[1]) for i in range(ackley.dimensions)]
    first_node = Individual.Individual(first_node_coordinates)
    # Add the first node to the solution path
    tabu_list.append([first_node, 0])
    current_node = first_node

    for i in range(max_iterations):
        neighborhood = find_neighborhood(current_node, tabu_list, ackley, neighborhood_size)
        next_node = select_next_node(neighborhood, ackley)
        score = ackley.function(next_node)
        tabu_list = add_node_tabu_list(tabu_list, tabu_list_size, next_node, i)
        current_node = next_node

        if score < best_score:
            best_score = score
            best_solution = next_node
        elif 0 <= abs(score-best_score) <= 1:
            # Chose a random first node
            first_node_coordinates = [random.uniform(ackley.bounds[0], ackley.bounds[1]) for i in range(ackley.dimensions)]
            first_node = Individual.Individual(first_node_coordinates)
            tabu_list.append([first_node, i])
            current_node = first_node

        tabu_list = update_tabu_list_ackley(tabu_list, tabu_time, i)
    best_score = ackley.function(best_solution)
    return best_solution, best_score


def find_neighborhood(current_node, tabu_list, ackley, neighborhood_size):
    neighborhood = []
    tabu_list_coordinates = [item[0].coordinates for item in tabu_list]
    sigma = 1.7

    for i in range(neighborhood_size):
        while True:
            perturbation = np.random.normal(loc=0.0, scale=sigma, size=ackley.dimensions)
            neighbor_coordinates = current_node.coordinates + perturbation
            if not neighbor_in_tabu(neighbor_coordinates, tabu_list_coordinates):
                neighborhood.append(Individual.Individual(neighbor_coordinates))
                break

    return neighborhood


def neighbor_in_tabu(neighbor_coordinates, tabu_list_coordinates):
    for node in tabu_list_coordinates:
        for i in range(len(node)):
            if node[i] == neighbor_coordinates[i]:
                return True
    return False


def select_next_node(neighborhood, ackley):
    function_list = [ackley.function(neighbor) for neighbor in neighborhood]
    min_neighbor_index = function_list.index(min(function_list))
    return neighborhood[min_neighbor_index]


def add_node_tabu_list(tabu_list, tabu_list_size, next_node, current_time):
    if len(tabu_list) == tabu_list_size:
        tabu_list.remove(tabu_list[0])
    tabu_list.append([next_node, current_time])
    return tabu_list


def update_tabu_list_ackley(tabu_list, tabu_time, current_time):
    update_tabu = tabu_list.copy()

    # removing individuals
    for i, ind in enumerate(tabu_list):
        if current_time - ind[1] >= tabu_time:
            update_tabu.remove(tabu_list[i])

    # update the time
    for ind in update_tabu:
        ind[1] += 1

    return update_tabu

