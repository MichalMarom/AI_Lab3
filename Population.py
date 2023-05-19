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
    best_individual: NQueensIndividual
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
            # print(f"========================================= {generation_index}")
            # print(f"Average for this gen is {new_average}")

            # Select the best individuals for reproduction
            elite_size = int(self.data.pop_size * ELITE_PERCENTAGE)
            elites = sorted(self.population, key=lambda NQueenInd: NQueenInd.score, reverse=True)[:elite_size] 

            # Generate new individuals by applying crossover and mutation operators
            offspring = []
            while len(offspring) < self.data.pop_size - elite_size:            
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                child_gen = []
                child_gen = self.cx_shuffle(parent1, parent2, self.gen_len)

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

            self.population = elites + offspring

            # Update the size of the  population
            self.data.pop_size = len(self.population)

        # Find the individual with the highest fitness
        self.best_individual = self.population[0]
        
        for individual in self.population:
            individual.update_score(self.data)
            if self.best_individual.score < individual.score:
                self.best_individual = individual

        self.best_fitness = abs(self.best_individual.score)
        return
    
    def get_solution(self):
        solution = []
        solution.append(self.start_point)
        
        for individual in self.best_individual.gen:
            solution.append(individual)

        solution.append(self.end_point)
        return solution

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

    def cx_shuffle(self, parent1: NQueensIndividual, parent2: NQueensIndividual, num_genes: int):
        p1 = parent1.gen
        p2 = parent2.gen

        cycles = [-1] * len(p1)
        cycle_no = 1
        cycle_start = (i for i, v in enumerate(cycles) if v < 0)

        for pos in cycle_start:

            while cycles[pos] < 0:
                cycles[pos] = cycle_no
                if p2[pos] in p1:
                    pos = p1.index(p2[pos])
                else:
                    pos = 0
            cycle_no += 1

        child_gen = [p1[i] if n % 2 else p2[i] for i, n in enumerate(cycles)]
        # [print(f"sol-> {ind.index}->") for ind in child_gen]

        return child_gen