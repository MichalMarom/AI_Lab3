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
    start_point: Individual
    end_point: Individual

    def __init__(self, population: list, 
                 center: Individual = None):

        self.individuals = population
        if center == None:
            self.update_center()
        else:
            self.center = center
        self.sum_demands = self.calc_sum_demands()
        self.score = self.calc_score()



    def calc_sum_demands(self):
        sum_demands = sum([individual.demand for individual in self.individuals])
        return sum_demands

    def calc_score(self):
        dist = 0
        for individual in self.individuals:
            dist += math.dist(individual.coordinates, self.center.coordinates)
        return dist

    def print_cluster(self):
        for i, individual in enumerate(self.individuals):
            print(f"the {i} individual coordinates is {individual.coordinates}, the weight is {individual.demand}, the index is {individual.index}")
        print(f"sum of demands is  {self.sum_demands}")
        print(f"The CENTER:  {self.center.coordinates}")

    def removing_individual(self, individual):
        individuals_coord = [ind.coordinates for ind in self.individuals]
        individual_index = individuals_coord.index(individual.coordinates)
        self.individuals.remove(self.individuals[individual_index])
        self.sum_demands = self.calc_sum_demands()
        self.update_center()

        return

    def adding_individual(self, individual):
        self.individuals.append(individual)
        self.sum_demands = self.calc_sum_demands()
        self.update_center()

        return

    def update_center(self):
        dist_individuals_list = []
        for i, individual in enumerate(self.individuals):
            dist = [math.dist(individual.coordinates, self.individuals[i].coordinates) for i in range(len(self.individuals))]
            dist_individuals_list.append((sum(dist)))

        min_individual_index = dist_individuals_list.index(min(dist_individuals_list))
        self.center = self.individuals[min_individual_index]


def best_fit(population: list, max_capacity: int):

    # Sort the items in decreasing order of size
    objects = [(i, ind.demand) for i, ind in enumerate(population)]
    objects = sorted(objects, key=lambda a: a[1], reverse=True)

    # Initialize the bins list with the first item
    bins = [[population[objects[0][0]]]]

    # Loop through the remaining items
    for item in objects[1:]:
        # Try to pack the item into an existing bin
        for bin in bins:
            sum_bin = sum([ind.demand for ind in bin])
            if sum_bin + item[1] <= max_capacity:
                bin.append(population[item[0]])
                break
        else:
            # If the item does not fit in any existing bin, create a new bin
            bins.append([population[item[0]]])
    return bins


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

    if not clusters_centers:
        clusters_centers = random.sample(population, k)

    for i in range(len(clusters_centers)):
        clusters.append([])

    for individual in population:
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
        nearest_individual = find_nearest_individual(cluster, expectation_center)
        new_clusters_centers.append(nearest_individual)

    return new_clusters_centers


def find_expectation_center(individuals: list):
    cluster_x_pr = [individual.coordinates[0] * (1 / len(individuals)) for individual in individuals]
    expectation_x = sum(cluster_x_pr)

    cluster_y_pr = [individual.coordinates[1] * (1 / len(individuals)) for individual in individuals]
    expectation_y = sum(cluster_y_pr)

    return [expectation_x, expectation_y]


def find_nearest_individual(individuals: list, expectation_center):
    dist = [math.dist(individual.coordinates, expectation_center) for individual in individuals]
    # new_center_index = dist.index(min(dist))
    # return individuals[new_center_index]

    min_dist_individuals = []
    for index, individual in enumerate(individuals):
        if dist[index] == min(dist):
            min_dist_individuals.append(individual)

    nearest_individual = random.sample(min_dist_individuals, 1)[0]

    return nearest_individual


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


