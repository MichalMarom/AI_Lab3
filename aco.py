# ----------- Python Package -----------
import math
import random

import numpy as np

# ----------- Consts Name  ----------
PHEROMONE = 50  # Amount of pheromone an ant borne with
ALPHA = 1   # The importance of the pheromone on the edge in the probabilistic transition
BETA = 1  # The importance of the length of the edge in the probabilistic transition
RHO = 0.8  # The trail persistence or evaporation rate


class Ant:
    pheromone: int
    current_node: list
    neighborhood: list
    edges_neighborhood: list
    path_nodes: list
    path_edges: list
    score: float

    def __init__(self, current_node, individuals, matrix_edges):
        self.pheromone = PHEROMONE
        self.current_node = current_node
        self.neighborhood = [ind for ind in individuals if ind.index != current_node.index]
        self.path_nodes = [current_node]
        self.path_edges = []
        self.update_edges(matrix_edges)
        self.score = 0

    # Updating the list of neighbors according to the nodes the ant have already visited
    def update_neighborhood(self):
        neighborhood_index = [neighbor.index for neighbor in self.neighborhood]
        for node in self.path_nodes:
            if node.index in neighborhood_index:
                self.neighborhood.remove(node)

    # Updating the list of edges according to the edges the ant have already visited
    def update_edges(self, matrix_edges):
        self.edges_neighborhood = []
        for neighbor in self.neighborhood:
            if self.current_node.index != neighbor.index:
                self.edges_neighborhood.append(matrix_edges[self.current_node.index][neighbor.index])

        edges_neighborhood_nodes = [edge.nodes for edge in self.edges_neighborhood]
        for edge in self.path_edges:
            if edge.nodes in edges_neighborhood_nodes:
                self.edges_neighborhood.remove(edge)
        return

    def select_next_node(self):
        importance_edges = [edge.tau ** ALPHA + edge.eta ** BETA for edge in self.edges_neighborhood]
        sum_importance_edges = sum(importance_edges)
        pro_select_edges = [importance_edges[i] / sum_importance_edges for i in range(len(self.edges_neighborhood))]
        next_index = pro_select_edges.index(max(pro_select_edges))
        next_edge = self.edges_neighborhood[next_index]
        next_node = next_edge.nodes[1]

        return next_node, next_edge

    # Adding the next edge and node to the ant path
    def update_path(self, next_node, next_edge):
        # Add the next node to the ant path
        self.path_nodes.append(next_node)
        self.path_edges.append(next_edge)

        # Update the current node that the ant reach
        self.current_node = next_node

        # Update the number of pheromone on the edge that the ant select
        next_edge.tau += PHEROMONE / next_edge.length

        return

    # Correcting the path so that it starts from point 0
    def fix_path_to_start(self, start_point):
        self.path_nodes.append(start_point)
        self.path_nodes = [start_point] + self.path_nodes
        first_edge = Edge([self.path_nodes[0], self.path_nodes[1]])
        last_edge = Edge([self.path_nodes[len(self.path_nodes) - 2], self.path_nodes[len(self.path_nodes) - 1]])
        self.path_edges.append(last_edge)
        self.path_edges = [first_edge] + self.path_edges

        return

    # Calculating and updating the length of the ant's path
    def update_score(self):
        dist = 0
        for i in range(len(self.path_nodes) - 1):
            dist += math.dist(self.path_nodes[i].coordinates, self.path_nodes[i+1].coordinates)
        self.score = dist
        return


class Edge:
    nodes: list
    length: float
    tau: float  # Intensity of the pheromone on the edge
    eta: float  # Heuristic function of the desirability of adding edge

    def __init__(self, nodes):
        self.nodes = nodes
        self.length = math.dist(nodes[0].coordinates, nodes[1].coordinates)
        self.tau = 0
        self.eta = 1 / math.dist(nodes[0].coordinates, nodes[1].coordinates)

    def update_tau(self):
        self.tau = (1-RHO)*self.tau + RHO*(PHEROMONE / self.length)


# ----------- Search path for each cluster -----------
def aco_algo(clusters, start_point):

    max_iterations = 100
    local_min = 0
    old_best_cost = 0
    num_ants = 5
    global ALPHA
    global BETA
    global RHO
    global PHEROMONE

    solution = []
    best_solution = []
    best_score = []

    for i in range(len(clusters)):
        solution.append([])
        best_solution.append([])
        best_score.append(float('inf'))

    matrix_clusters_edges = [init_matrix_edges(cluster.individuals) for cluster in clusters]

    for iteration in range(max_iterations):
        for i, cluster in enumerate(clusters):
            # nodes = random.sample(cluster.individuals, num_ants)
            first_node = random.sample(cluster.individuals, 1)[0]

            # Saving the matrix of edges within the cluster
            matrix_edges = matrix_clusters_edges[i]

            # Put the ants at the randoms nodes
            ants = [Ant(first_node, cluster.individuals, matrix_edges) for i in range(num_ants)]

            # The path start the location of the ant
            path_length = 0
            while path_length < len(cluster.individuals) - 1:
                for ant in ants:
                    ant.update_neighborhood()
                    ant.update_edges(matrix_edges)
                    next_node, next_edge = ant.select_next_node()
                    ant.update_path(next_node, next_edge)
                path_length += 1

            for ant in ants:
                ant.fix_path_to_start(start_point)
                ant.update_score()
                # print(f"{[node.index for node in ant.path_nodes]} , {ant.score}")

            path, score = find_best_path(ants)
            if score < best_score[i]:
                best_score[i] = score
                best_solution[i] = path

        new_best_cost = sum(best_score)
        if old_best_cost == new_best_cost:
            local_min += 1
            if local_min == 4:
                update_edges(matrix_edges, clusters[i].individuals, increasing_explortion=True)
                local_min = 0
                # num_ants += 2

        old_best_cost = new_best_cost

        for i, matrix_edges in enumerate(matrix_clusters_edges):
            update_edges(matrix_edges, clusters[i].individuals,  increasing_explortion=False)

    return best_solution, sum(best_score)


# Initialize the edges matrix for a cluster
def init_matrix_edges(individuals):
    individuals_index = [ind.index for ind in individuals]
    matrix_size = max(individuals_index)+1
    matrix_edges = []

    # Create List of list
    for i in range(matrix_size):
        matrix_edges.append([])
        for j in range(matrix_size):
            matrix_edges[i].append(None)

    for i_ind in individuals:
        for j_ind in individuals:
            if i_ind.index != j_ind.index:
                nodes = [i_ind, j_ind]
                matrix_edges[i_ind.index][j_ind.index] = Edge(nodes)

    return matrix_edges


# Returns the shortest path and it's score
def find_best_path(ants):
    scores = [ant.score for ant in ants]
    min_scores_index = scores.index(min(scores))
    min_path = ants[min_scores_index].path_nodes
    return min_path, min(scores)


def find_best_path(ants):
    scores = [ant.score for ant in ants]
    min_scores_index = scores.index(min(scores))
    min_path = ants[min_scores_index].path_nodes
    return min_path, min(scores)


# Update the tau for each edge in the cluster
def update_edges(matrix_edges, individuals, increasing_explortion: bool):
    for i_ind in individuals:
        for j_ind in individuals:
            if i_ind.index != j_ind.index:
                # if increasing_explortion and matrix_edges[i_ind.index][j_ind.index].tau < 10:
                #     matrix_edges[i_ind.index][j_ind.index].tau *= 10
                if increasing_explortion:
                    matrix_edges[i_ind.index][j_ind.index].tau = 0.1
                else:
                    matrix_edges[i_ind.index][j_ind.index].update_tau()
    return

