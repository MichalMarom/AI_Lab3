# ----------- File Form Lab -----------
import random
import Data
import Individual
import Clustering
# ----------- Python Package -----------
import numpy as np
import math
# ----------- Consts Parameters -----------
# ----------- Consts Name  -----------


class Population:
    data: Data
    individuals: list
    start_point: Individual
    max_capacity : int
    supermarket_number: int
    trucks_number: int
    clusters: list
    total_score: float

    def __init__(self, setting_vector = None):
        self.data = Data.Data(setting_vector)
        self.individuals = []
        self.start_point = None
        self.clusters = []
        self.read_problem_file()

    def read_problem_file(self):
        with open("CVRP_inputs/cvpr_test1.txt") as f:
            f.readline()
            # Numbers of optimal trucks
            fields = f.readline().split(" ")
            chars = [x for x in fields[9]]
            self.trucks_number = int(chars[0])

            f.readline()
            # Numbers of supermarket
            fields = f.readline().split(" ")
            self.supermarket_number = int(fields[2])

            f.readline()
            # Max capacity for a truck
            fields = f.readline().split(" ")
            self.max_capacity = int(fields[2])

            f.readline()
            # Initialize coordinates of each supermarket
            coordinates_dict = {}
            for i in range(self.supermarket_number):
                fields = f.readline().split(" ")
                coordinates_dict[i] = [int(fields[1]), int(fields[2])]

            f.readline()
            # Initialize demand of each supermarket
            demands_dict = {}
            for i in range(self.supermarket_number):
                fields = f.readline().split(" ")
                demands_dict[i] = int(fields[1])

            self.create_individuals(coordinates_dict, demands_dict)
            self.start_point = self.individuals[0]
            self.individuals.remove(self.individuals[0])
            return

    def create_individuals(self, coordinates_dict, demands_dict):
        for i in range(self.supermarket_number):
            ind = Individual.Individual(coordinates_dict[i], demands_dict[i])
            self.individuals.append(ind)
        return

    def print_pop(self):
        print(f"num of trucks is {self.trucks_number}")
        print(f"num of max capacity is {self.max_capacity}")
        print(f"the starting point coordinates are: {self.start_point.coordinates}")
        for i, individual in enumerate(self.individuals):
            print(f"the {i} individual coordinates are {individual.coordinates}, and the weight is {individual.demand}")
        return

    def print_clusters(self):
        for i, cluster in enumerate(self.clusters):
            print(f"the {i} cluster is :")
            cluster.print_cluster()
            print("=================================")
        return

    def create_clusters(self):
        clusters_centers, clusters = Clustering.clustering(self.individuals, self.trucks_number)
        for i in range(len(clusters)):
            self.clusters.append(Clustering.Cluster(clusters[i], clusters_centers[i]))
        return


