# ----------- File Form Lab -----------
import Data
import Individual
import Clustering
import TabuSearch
import aco
# ----------- Python Package -----------
import random
import math
import matplotlib.pyplot as plt
# ----------- Consts Parameters -----------
# ----------- Consts Name  -----------
Tabu_search = 0
ACO = 1
Simulated_Annealing = 2
GA = 3
Cooperative_PSO = 4
MAX_TRY = 5


class Population:
    data: Data
    individuals: list
    start_point: Individual
    max_capacity: int
    supermarket_number: int
    trucks_number: int
    clusters: list
    total_score: float
    solution: list

    def __init__(self, setting_vector=None):
        self.data = Data.Data(setting_vector)
        self.individuals = []
        self.start_point = None
        self.clusters = []
        self.total_score = 0
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

            # Initialize each supermarket by coordinates and demands
            self.create_individuals(coordinates_dict, demands_dict)

            return

    def create_individuals(self, coordinates_dict, demands_dict):
        for i in range(self.supermarket_number):
            if i == 0:
                self.start_point = Individual.Individual(coordinates_dict[i], demands_dict[i], i)
            else:
                ind = Individual.Individual(coordinates_dict[i], demands_dict[i], i)
                self.individuals.append(ind)
        return

    def print_pop(self):
        print(f"num of trucks is {self.trucks_number}")
        print(f"num of max capacity is {self.max_capacity}")
        print(f"the starting point coordinates are: {self.start_point.coordinates}")

        for i, individual in enumerate(self.individuals):
            print(f"the {i} individual coordinates are {individual.coordinates}, and the weight is {individual.demand}")
        print("===================================================================================================")
        return

    def print_clusters(self):
        for i, cluster in enumerate(self.clusters):
            print(f"the {i} cluster is :")
            cluster.print_cluster()
            print("=================================")
        return

    def print_graph(self):
        x1 = []
        y1 = []
        colors = []
        ax = plt.axes()
        ax.set(xlim=(0, 1000),
               ylim=(0, 1000),
               xlabel='X',
               ylabel='Y')

        for j in range(len(self.clusters)):
            rand_colors = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            colors.append(rand_colors)

        # ----- Print Clusters -----
        # for i, cluster in enumerate(self.clusters):
        #     for individual in cluster.individuals:
        #         x1.append(individual.coordinates[0])
        #         y1.append(individual.coordinates[1])
        #         ax.annotate(individual.index, (individual.coordinates[0], individual.coordinates[1]))
        #     plt.plot(x1, y1, color=colors[i],  marker='o')
        #     plt.plot(cluster.center.coordinates[0], cluster.center.coordinates[1], color='red', marker='o')
        #     x1 = []
        #     y1 = []

        x1.append(self.start_point.coordinates[0])
        y1.append(self.start_point.coordinates[1])
        ax.annotate(0, (self.start_point.coordinates[0], self.start_point.coordinates[1]))
        plt.plot(x1, y1, color='red',  marker='o')
        x1 = []
        y1 = []

        for i, path in enumerate(self.solution):
            for point in path:
                x1.append(point.coordinates[0])
                y1.append(point.coordinates[1])
                ax.annotate(point.index, (point.coordinates[0], point.coordinates[1]))
            plt.plot(x1, y1, color=colors[i],  marker='o')
            x1 = []
            y1 = []
        plt.show()
        return

    def create_clusters(self):

        # ------ Clustring with best fit - by weight ------
        # clusters = Clustering.best_fit(self.individuals, self.max_capacity)
        # for i in range(len(clusters)):
        #     self.clusters.append(Clustering.Cluster(clusters[i]))

        # ------ Clustring KNN - by dist ------
        # clusters_centers, clusters = Clustering.clustering(self.individuals, self.trucks_number)
        # for i in range(len(clusters)):
        #     self.clusters.append(Clustering.Cluster(clusters[i], clusters_centers[i]))
        #
        # while True:
        #     clusters_valid_check = [cluster.sum_demands > self.max_capacity for cluster in self.clusters]
        #     print(clusters_valid_check)
        #     if True not in clusters_valid_check:
        #         break
        #     elif True in clusters_valid_check:
        #         self.fix_cluster_weight()

        while True:
            trys = 0
            self.clusters = []
            clusters_centers, clusters = Clustering.clustering(self.individuals, self.trucks_number)
            for i in range(len(clusters)):
                self.clusters.append(Clustering.Cluster(clusters[i], clusters_centers[i]))

            while trys < MAX_TRY:
                clusters_valid_check = [cluster.sum_demands > self.max_capacity for cluster in self.clusters]
                print(clusters_valid_check)
                if True not in clusters_valid_check:
                    print("break INER loop")
                    break
                elif True in clusters_valid_check:
                    self.fix_cluster_weight()
                    trys += 1
            else:
                print("continue")
                continue
            print("break OYTER loop")
            break

        return

    def fix_cluster_weight(self):
        for index, cluster in enumerate(self.clusters):
            print("AT INDEX:", index)
            if cluster.sum_demands > self.max_capacity:
                self.balance_cluster_weight(cluster, index)

        return

    def balance_cluster_weight(self, cluster, cluster_index):

        # ------ balance with: Dist + weight ------
        # if cluster_index == len(self.clusters)-1:
        #     dist = [math.dist(cluster.center.coordinates, self.clusters[i].center.coordinates) for i in range(len(self.clusters)-1)]
        #     weight = [self.clusters[i].sum_demands for i in range(len(self.clusters) - 1)]
        #     clusters_score = [dist[i] + weight[i] for i in range(len(self.clusters) - 1)]
        #     # print("len dist:", len(dist))
        #     min_centers = []
        #     for i in range(len(clusters_score)):
        #         if clusters_score[i] == min(clusters_score):
        #             min_centers.append(self.clusters[i])
        #     closest_cluster = random.sample(min_centers, 1)[0]
        #
        #
        # else:
        #     dist = [math.dist(cluster.center.coordinates, self.clusters[i].center.coordinates) for i in range(cluster_index+1, len(self.clusters))]
        #     weight = [self.clusters[i].sum_demands for i in range(cluster_index+1, len(self.clusters))]
        #     clusters_score = [dist[i] + weight[i] for i in range(len(dist))]
        #     # print("len dist:", len(dist))
        #     min_centers = []
        #     for i in range(len(clusters_score)):
        #         if clusters_score[i] == min(clusters_score):
        #             min_centers.append(self.clusters[i + cluster_index + 1])
        #     closest_cluster = random.sample(min_centers, 1)[0]

        # ------ balance with: Dist + weight, with circles ------
        dist = [math.dist(cluster.center.coordinates, self.clusters[i].center.coordinates) for i in range(len(self.clusters))]
        weight = [self.clusters[i].sum_demands for i in range(len(self.clusters))]
        clusters_score = [0.5*dist[i] + 0.5*weight[i] for i in range(len(self.clusters))]
        min_centers = []
        for i in range(len(clusters_score)):
            if clusters_score[i] == min(clusters_score) and i != cluster_index:
                min_centers.append(self.clusters[i])
        closest_cluster = random.sample(min_centers, 1)[0]

        # ------ balance with: Dist only ------
        # For the last cluster lets choose according to weight or Dist
        # if cluster_index == len(self.clusters)-1:
        #     dist = [math.dist(cluster.center.coordinates, self.clusters[i].center.coordinates) for i in range(len(self.clusters)-1)]
        #     weight = [self.clusters[i].sum_demands for i in range(len(self.clusters) - 1)]
        #     # print("len dist:", len(dist))
        #     min_dist_centers = []
        #     for i in range(len(dist)):
        #         if dist[i] == min(dist):
        #             min_dist_centers.append(self.clusters[i])
        #     closest_cluster = random.sample(min_dist_centers, 1)[0]
        #
        # else:
        #     dist = [math.dist(cluster.center.coordinates, self.clusters[i].center.coordinates) for i in range(cluster_index+1, len(self.clusters))]
        #     # print("len dist:", len(dist))
        #     min_dist_centers = []
        #     for i in range(len(dist)):
        #         if dist[i] == min(dist):
        #             min_dist_centers.append(self.clusters[i + cluster_index + 1])
        #     closest_cluster = random.sample(min_dist_centers, 1)[0]

        while cluster.sum_demands > self.max_capacity:
            nearest_individual = self.find_nearest_individual(closest_cluster, cluster)
            # Removing the nearest individual from cluster and updating sum_demands
            cluster.removing_individual(nearest_individual)

            # Adding the nearest individual to the closest cluster and updating sum_demands
            closest_cluster.adding_individual(nearest_individual)

        print("finish!")
        return

    def find_nearest_individual(self, closest_cluster, cluster):
        dist = [math.dist(individual.coordinates, closest_cluster.center.coordinates) for individual in cluster.individuals]
        nearest_individual_index = dist.index(min(dist))
        return cluster.individuals[nearest_individual_index]

    def solve_clustrers_TSP(self):
        self.solution = []
        # if self.data.algorithm == Tabu_search:
        self.solution, self.total_score = TabuSearch.tabu_search(self.clusters, self.start_point)
        # for i, path in enumerate(self.solution):
        #     dist = 0
        #     for j in range(len(path) - 1):
        #         dist += math.dist(path[j].coordinates, path[j + 1].coordinates)
        #     self.clusters[i].score = dist
        #     self.total_score += dist

        #if self.data.algorithm == ACO:
        # self.solution = aco.aco_algo(self.clusters, self.start_point)
        # for i, path in enumerate(self.solution):
        #     dist = 0
        #     for j in range(len(path) - 1):
        #         dist += math.dist(path[j].coordinates, path[j + 1].coordinates)
        #     self.clusters[i].score = dist
        #     self.total_score += dist

        for path in self.solution:
            print("----------------")
            for point in path:
                print(point.index)
        print("TOTAL SCORE: ", self.total_score)

        return self.solution, self.total_score



