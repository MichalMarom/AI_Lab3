# ----------- File For Genetic Algorithm -----------
from Clustering import Cluster
import Data
import NQueensIndividual
import Individual
# ----------- Python Package -----------
import time
import numpy as np
import random
import matplotlib.pyplot as plt
# ----------- Consts Parameters -----------
MUTATION_INDIVIDUALS = 20
ELITE_PERCENTAGE = 0.20
# ----------- Consts Name  -----------
STRING = 0
N_QUEENS = 1
BIN_PACKING = 2
CARTESIAN = 3
SHARED_FIT = 0
CLUSTER = 1
CROWDING = 2
CONSTRAINT_1 = 1
CONSTRAINT_2 = 2


class Population:
    data: Data
    population: list
    score: float
    start_point: Individual
    end_point: Individual
    gen_len: int

    best_fitness: float
    best_individual: Individual
    center: Individual
    optimization_func: int
    fitnesses: list

    def __init__(self, cluster: Cluster, start_point, end_point):
        
        self.gen_len = len(cluster.individuals)
        self.data = Data.Data(setting_vector = self.gen_len)
        self.population = []
        self.score = 0
        self.start_point = start_point
        self.end_point = end_point

        self.fitnesses = []
        self.best_individual = None
        self.best_fitness = 0
        self.max_weight = 0
        self.objects = []  

        for index in range(self.data.pop_size):
            individual = NQueensIndividual.NQueensIndividual(self.data, 
                                           cluster.individuals, 
                                           self.start_point, 
                                           self.end_point)
            self.population.append(individual)
            self.set_fitnesses()
        
        print(f"population len-> {len(self.population)}")
        print(f"individuals len-> {len(self.population[0].gen)}")
        
        print(f"the first individual-> {self.population[0].gen[0].coordinates}")
        return
    
    def set_fitnesses(self):
        self.fitnesses = []
        for individual in self.population:
            self.fitnesses.append(individual.score)
        return
    
    def genetic_algorithm(self):
        for generation_index in range(self.data.max_generations):
            mutation_individuals = MUTATION_INDIVIDUALS

            old_average, old_variance, old_sd = self.average_fitness(self.fitnesses)
            for index, individual in enumerate(self.population):
                self.fitnesses[index] = individual.score

            new_average, new_variance, new_sd = self.average_fitness(self.fitnesses)

            gen_time = time.time()
            print(f"========================================= {generation_index}")
            print(f"fitnesses for this gen is {self.fitnesses}")
            print(f"Average for this gen is {new_average}")
            print(f"Selection Pressure for this gen is {new_variance}")
            # self.show_histogram(self.fitnesses)

            # Select the best individuals for reproduction
            elite_size = int(self.data.pop_size * ELITE_PERCENTAGE)
            elite_indices = sorted(range(self.data.pop_size), key=lambda i: self.fitnesses[i], reverse=True)[:elite_size]
            elites = [self.population[i] for i in elite_indices]

            # Generate new individuals by applying crossover and mutation operators
            offspring = []
            while len(offspring) < self.data.pop_size - elite_size:            
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                child_gen = []
                ran = random.random()
                copy_objects = parent1.gen.copy()

                parent1_part = int(len(copy_objects)*ran)
                parent2_part = len(copy_objects) - parent1_part

                for i in range(parent1_part):
                    object = random.sample(copy_objects, 1)[0]
                    if object in parent1.gen:
                        child_gen.append(object)
                        copy_objects.remove(object)
                        break

                for i in range(parent2_part):
                    object = random.sample(copy_objects, 1)[0]
                    if object in parent2.gen:
                        child_gen.append(object)
                        copy_objects.remove(object)
                        break
                
                child = NQueensIndividual.NQueensIndividual(self.data,
                                                            child_gen,
                                                            self.start_point, 
                                                            self.end_point)              

                child.gen = child_gen
                child.gen_len = len(child_gen)
                child.update_score(self.data)
                
                # mutation
                if new_average == old_average and mutation_individuals > 0:
                    child.mutation(self.data)
                    child.update_score(self.data)
                    mutation_individuals -= 1

                offspring.append(child)
            # print(f"the elites is -> {elites}")
            # print(f"the offspring is -> {offspring}")
            # temp = []
            # temp = [temp.append(child.gen[i].index) for i in range(child.gen_len)]
            # print(f"the gen is -> {temp}")
            # print(f"the score is -> {child.score}")

            self.population = elites + offspring

            # Update the size of the  population
            self.data.pop_size = len(self.population)

            print(f"The absolute time for this gen is {time.time() - gen_time} sec")
            print(f"The ticks time for this gen is {int(time.perf_counter())}")




        # Find the individual with the highest fitness
        self.best_individual = self.population[0]
        
        for individual in self.population:
            individual.update_score(self.data)
            if self.best_individual.score < individual.score:
                self.best_individual = individual
                print("changed best individual------------------------------")

        self.best_fitness = self.best_individual.score

        # print(f"population len-> {len(self.population)}")
        # print(f"best individual len-> {len(self.best_individual.gen)}")
        # print(f"individuals len-> {len(self.population[0].gen)}")



        # print(f"sol-> {self.best_individual.gen}")
        # print(f"score-> {self.best_fitness}")
        return
    
    def average_fitness(self, fitness: list): 
        if not fitness:
            return 0
        try:
            average = sum(fitness) / len(fitness)
            variance = sum([((x - average) ** 2) for x in fitness]) / (len(fitness) - 1)
        except:
            average = 0
            variance = 0
        sd = variance ** 0.5

        return average, variance, sd

    def show_histogram(self, array):
        np_array = np.array(array)
        plt.hist(np_array)
        plt.show()
        return
