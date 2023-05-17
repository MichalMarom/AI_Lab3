# ----------- File Form Lab -----------
import threading
import Data
import Individual
import Clustering
import SimulatedAnnealing
import TabuSearch
import aco
import CooperativePSO
import Population
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
MAX_TRY_CLUSTER = 100

TABU_SEARCH = 1
SIMULATED_ANNEALING = 2
ISLANDS = 3


class CVRP:
    data: Data
    max_capacity: int
    supermarket_number: int
    trucks_number: int

    individuals: list
    start_point: Individual
    end_point: Individual

    clusters: list
    islands: list
    
    total_score: float
    solution: list

    def __init__(self, setting_vector=None):
        self.data = Data.Data(setting_vector)
        self.individuals = []
        # self.start_point = Individual.Individual([0,0], 0, 0)
        # self.end_point = Individual.Individual([0,0], 0, len(self.individuals))
        self.clusters = []
        self.islands = []
        self.total_score = 0
        self.solution = []
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
            
            self.end_point = Individual.Individual(self.start_point.coordinates, 0, len(self.individuals)+1)

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
        x_values = []
        y_values = []
        for point in self.individuals:
            x_values.append(point.coordinates[0])
            y_values.append(point.coordinates[1])
        
        max_value_x = max(x_values) + 10
        max_value_y = max(y_values) + 10
        min_value_x = min(x_values) - 10
        min_value_y = min(y_values) - 10
        x1 = []
        y1 = []
        colors = []
        ax = plt.axes()
        ax.set(xlim=(min_value_x, max_value_x),
               ylim=(min_value_y, max_value_y),
               xlabel='X',
               ylabel='Y')

        for j in range(len(self.clusters)):
            rand_colors = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            colors.append(rand_colors)

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
            # plt.plot([self.clusters[i].center.coordinates[0]], [self.clusters[i].center.coordinates[1]], color='red',marker='X')
            x1 = []
            y1 = []
        plt.show()
        return

    def create_clusters(self):

        # ------ Clustring with best fit - by weight ------
        # clusters = Clustering.best_fit(self.individuals, self.max_capacity)
        # for i in range(len(clusters)):
        #     self.clusters.append(Clustering.Cluster(clusters[i]))
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
        silhouette_per_try = []
        inertia_per_try = []
        clusters_per_try = []
        for i in range(MAX_TRY_CLUSTER):
            while True:
                trys = 0
                self.clusters = []
                clusters_centers, clusters = Clustering.clustering(self.individuals, self.trucks_number)
                for i in range(len(clusters)):
                    self.clusters.append(Clustering.Cluster(clusters[i], clusters_centers[i]))

                while trys < MAX_TRY:
                    clusters_valid_check = [cluster.sum_demands > self.max_capacity for cluster in self.clusters]
                    if True not in clusters_valid_check:
                        break
                    elif True in clusters_valid_check:
                        self.fix_cluster_weight()
                        trys += 1
                else:
                    continue
                break

            clusters_centers = [cluster.center for cluster in self.clusters]
            silhouette_score = Clustering.silhouette(clusters_centers, self.clusters)
            silhouette_per_try.append(silhouette_score)
            inertia_score = Clustering. inertia(clusters_centers, self.clusters)
            inertia_per_try.append(inertia_score)
            clusters_per_try.append(self.clusters)

        self.clusters = Clustering.find_best_cluster(clusters_per_try, silhouette_per_try)
        return

    def fix_cluster_weight(self):
        for index, cluster in enumerate(self.clusters):
            # print("AT INDEX:", index)
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

        # print("finish!")
        return

    def find_nearest_individual(self, closest_cluster, cluster):
        dist = [math.dist(individual.coordinates, closest_cluster.center.coordinates) for individual in cluster.individuals]
        nearest_individual_index = dist.index(min(dist))
        return cluster.individuals[nearest_individual_index]

    def solve_clustrers_TSP(self, algorithm_type):
        if algorithm_type == Tabu_search:
            self.solve_with_tabu_search()

        elif algorithm_type == ACO:
            self.solve_with_aco()

        elif algorithm_type == Simulated_Annealing:
            self.solve_with_simulated_anealing()

        elif algorithm_type == GA:
            self.solve_with_islands_genetic_algo()

        elif algorithm_type == Cooperative_PSO:
            self.solve_with_Cooperative_PSO()

        return

    def solve_with_tabu_search(self):
        self.solution, self.total_score = TabuSearch.tabu_search(self.clusters, self.start_point)
        print("TOTAL SCORE: ", self.total_score)
        return

    def solve_with_aco(self):
        self.solution, self.total_score = aco.aco_algo(self.clusters, self.start_point)
        print("TOTAL SCORE: ", self.total_score)
        return

    def solve_with_simulated_anealing(self):

        for cluster in self.clusters:
            simulated_annealing_instance = SimulatedAnnealing.SimulatedAnnealing(cluster,
                                                                                 self.start_point,
                                                                                 self.end_point)
            simulated_annealing_instance.simulated_annealing()
            solution, score = simulated_annealing_instance.get_solution_and_socre()
            self.solution.append(solution)
            self.total_score += score

        # for path in self.solution:
        #     print("----------------")
        #     for point in path:
        #         print(point.index)
        print("TOTAL SCORE: ", int(self.total_score))

        return

    def solve_with_Cooperative_PSO(self):
        self.solution, self.total_score = CooperativePSO.cooperative_pso(self.clusters, self.start_point)
        print("TOTAL SCORE: ", self.total_score)
        return
    
    def solve_with_islands_genetic_algo(self):
        # Initialize the cvrp for the current island
        for i, cluster in enumerate(self.clusters):
            self.islands.append(Population.Population(cluster, self.start_point, self.end_point))
        
        # Create and start threads for each island
        threads = []
        for i, island in enumerate(self.islands):
            thread = threading.Thread(target = self.islands[i].genetic_algorithm, 
                                      args=())
            threads.append(thread)
        
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        for i, island in enumerate(self.islands):
            self.solution.append(self.islands[i].get_solution())
            self.total_score += island.best_fitness

        for path in self.solution:
            print("----------------")
            for point in path:
                print(point.index)

        print("TOTAL SCORE: ", int(self.total_score))

        return


