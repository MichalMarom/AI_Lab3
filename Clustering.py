# ----------- File Form Lab -----------
import Individual
# ----------- Python Package -----------
import math
import random


class Cluster:
    individuals: list
    center: Individual
    sum_demands: int
    score: float

    def __init__(self, pop: list, center: Individual):
        self.individuals = pop
        self.center = center
        self.sum_demands = sum([individual.demand for individual in self.individuals])
        self.score = self.calc_score()

    def calc_score(self):
        dist = 0
        for individual in self.individuals:
            dist += math.dist(individual.coordinates, self.center.coordinates)
        return dist

    def print_cluster(self):
        for i, individual in enumerate(self.individuals):
            print(f"the {i} individual coordinates is {individual.coordinates}, and the weight is {individual.demand}")


def clustering(population: list, k: int):
    clusters_centers_update = []
    while True:
        clusters_centers_previous, clusters_previous = knn(k, population, clusters_centers_update)
        clusters_centers_update = update_clusters_centers(clusters_previous)
        clusters_centers_update, clusters_update = knn(k, population, clusters_centers_update)
        if equal_centers(clusters_centers_previous, clusters_centers_update):
            break

    return clusters_centers_update, clusters_update


def knn(k: int, population: list, clusters_centers: list):
    clusters = []
    clusters_centers_gen = []

    if not clusters_centers:
        clusters_centers = random.sample(population, k)

    for i in range(len(clusters_centers)):
        clusters.append([])

    for individual in population:
        if individual.coordinates not in clusters_centers_gen:
            dist = [math.dist(individual.coordinates, center.coordinates) for center in clusters_centers]
            min_dist_centers = []
            for index, center in enumerate(clusters_centers):
                if dist[index] == min(dist):
                    min_dist_centers.append((index, center))

            closest_center = random.sample(min_dist_centers, 1)[0]
            clusters[closest_center[0]].append(individual)

    return clusters_centers, clusters


def update_clusters_centers(clusters: list):
    new_clusters_centers = []

    for i, cluster in enumerate(clusters):
        expectation_center = find_expectation_center(cluster)
        new_clusters_centers.append(find_nearest_individual(cluster, expectation_center))

    return new_clusters_centers


def find_expectation_center(individuals: list):
    cluster_x_pr = [individual.coordinates[0] * (1 / len(individuals)) for individual in individuals]
    expectation_x = sum(cluster_x_pr)

    cluster_y_pr = [individual.coordinates[1] * (1 / len(individuals)) for individual in individuals]
    expectation_y = sum(cluster_y_pr)

    return [expectation_x, expectation_y]


def find_nearest_individual(individuals: list, expectation_center):
    dist = [math.dist(individual.coordinates, expectation_center) for individual in individuals]
    new_center_index = dist.index(min(dist))

    return individuals[new_center_index]


def equal_centers(clusters_centers_previous: list, clusters_centers_update: list):
    coordinates_clusters_centers_previous = [individual.coordinates for individual in clusters_centers_previous]
    coordinates_clusters_centers_update = [individual.coordinates for individual in clusters_centers_update]

    for coordinates in coordinates_clusters_centers_previous:
        if coordinates not in coordinates_clusters_centers_update:
            return False

    for coordinates in coordinates_clusters_centers_update:
        if coordinates not in coordinates_clusters_centers_previous:
            return False

    return True


