# ----------- Python Package -----------
import math
import random

import numpy as np

# ----------- Consts Name  ----------
PHEROMONE = 50  # Amount of pheromone an ant borne with
ALPHA = 1  # The importance of the pheromone on the edge in the probabilistic transition
BETA = 1  # The importance of the length of the edge in the probabilistic transition
RHO = 0.2  # The trail persistence or evaporation rate


class Ant:
    pheromone: int
    current_node: list
    neighborhood: list
    edges_neighborhood: list
    path_nodes: list
    path_edges: list
    score: float

    def __init__(self, current_node, start_point, individuals, matrix_edges):
        self.pheromone = PHEROMONE
        self.current_node = current_node
        self.neighborhood = [ind for ind in individuals if ind.coordinates != current_node.coordinates]
        self.edges_neighborhood = []
        self.path_nodes = [start_point, current_node]

        first_edge = Edge([start_point, current_node])
        first_edge.tau += PHEROMONE / first_edge.length
        self.path_edges = [first_edge]

        self.update_edges(matrix_edges)
        self.score = 0

    def update_neighborhood(self):
        neighborhood_coord = [neighbor.coordinates for neighbor in self.neighborhood]
        for node in self.path_nodes:
            if node.coordinates in neighborhood_coord:
                self.neighborhood.remove(node)
        # print("neighborhood:", [node.index for node in self.neighborhood])

    def update_edges(self, matrix_edges):
        self.edges_neighborhood = []
        for neighbor in self.neighborhood:
            self.edges_neighborhood.append(matrix_edges[self.current_node.index-1][neighbor.index-1])

        edges_neighborhood_nodes = [edge.nodes for edge in self.edges_neighborhood]
        for edge in self.path_edges:
            if edge.nodes in edges_neighborhood_nodes:
                self.edges_neighborhood.remove(edge)


        return

    def select_next_node(self):
        importance_edges = [edge.tau ** ALPHA + edge.eta ** BETA for edge in self.edges_neighborhood]
        sum_importance_edges = sum(importance_edges)
        pro_select_edges = [importance_edges[i] / sum_importance_edges for i in range(len(self.edges_neighborhood))]
        if not pro_select_edges:
            print("PROBLEM")
        next_index = pro_select_edges.index(max(pro_select_edges))
        next_edge = self.edges_neighborhood[next_index]
        next_node = next_edge.nodes[1]

        return next_node, next_edge

    def update_path(self, next_node, next_edge):
        # Add the next node to the ant path
        self.path_nodes.append(next_node)
        self.path_edges.append(next_edge)

        # Update the current node that the ant reach
        self.current_node = next_node

        # Update the number of pheromone on the edge that the ant select
        next_edge.tau += PHEROMONE / next_edge.length

        return

    def update_score(self):
        dist = 0
        for i in range(len(self.path_nodes) - 1):
            dist += math.dist(self.path_nodes[i].coordinates, self.path_nodes[i+1].coordinates)
        self.score = dist


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

    def print_data(self):
        print([self.nodes[0].index, self.nodes[1].index])


def aco_algo(clusters, start_point):

    max_iterations = 10
    best_solution = []
    best_solution_score = float('inf')
    matrix_clusters_edges = [init_matrix_edges(cluster.individuals) for cluster in clusters]
    num_of_ants = 0

    for iteration in range(max_iterations):
        print(f"======iteration: {iteration}======")
        solution = []
        score = 0

        for i, cluster in enumerate(clusters):
            # nodes = random.sample(cluster.individuals, len(cluster.individuals))
            if iteration == 0:
                num_of_ants = len(cluster.individuals)

            nodes = random.sample(cluster.individuals, num_of_ants)
            matrix_edges = matrix_clusters_edges[i]
            ants = [Ant(node, start_point, cluster.individuals, matrix_edges) for node in nodes]
            path_length = 2
            while path_length < len(cluster.individuals)+1:
                for ant in ants:
                    ant.update_neighborhood()
                    ant.update_edges(matrix_edges)
                    next_node, next_edge = ant.select_next_node()
                    ant.update_path(next_node, next_edge)
                path_length += 1

            for ant in ants:
                ant.update_score()

            best_path, best_score = find_best_path(ants)
            score += best_score
            solution.append(best_path)

        print("score: ", score)

        if score < best_solution_score:
            best_solution_score = score
            best_solution = solution

        for i, matrix_edges in enumerate(matrix_clusters_edges):
            update_edges(matrix_edges, clusters[i].individuals)

    return best_solution


def update_edges(matrix_edges, individuals):
    for i_ind in individuals:
        for j_ind in individuals:
            if i_ind.index != j_ind.index:
                matrix_edges[i_ind.index-1][j_ind.index-1].update_tau()

    return


def init_matrix_edges(individuals):
    individuals_index = [ind.index for ind in individuals]
    matrix_size = max(individuals_index)
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
                matrix_edges[i_ind.index - 1][j_ind.index - 1] = Edge(nodes)

    return matrix_edges


def find_best_path(ants):
    scores = [ant.score for ant in ants]
    min_scores_index = scores.index(min(scores))
    min_path = ants[min_scores_index].path_nodes
    return min_path, min(scores)
