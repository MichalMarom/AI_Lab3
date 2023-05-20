# ----------- File For Genetic Algorithm -----------
from Clustering import Cluster
import Data
import NQueensIndividual
import Individual
# from Ackley import AckleyFunction
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
    

# ----------------------- Not a class method --------------------
def solve_ackley(ackley):
    # score: float = 0
    pop_size: int = 100
    dimensions: int = ackley.dimensions
    max_generations: int = 100

    population: list = []
    fitnesses: list = []

    # create population
    for index in range(pop_size):
        first_node_coordinates = [random.uniform(ackley.bounds[0], ackley.bounds[1]) for i in range(dimensions)]
        individual = Individual.Individual(first_node_coordinates)
        individual.score = ackley.function(individual)
        population.append(individual)

    #create fitnesses
    for individual in population:
        fitnesses.append(individual.score)

    scores = []

    #starting GA
    for generation_index in range(max_generations):
        for index, individual in enumerate(population):
            fitnesses[index] = ackley.function(individual)

        average_fitness = np.average(fitnesses)
        # gen_time = time.time()
        # print(f"========================================= {generation_index}")
        # print(f"Average for this gen is {average_fitness}")

        # Select the best individuals for reproduction
        elite_size = int(pop_size * ELITE_PERCENTAGE)
        elites = sorted(population, key=lambda individual: individual.score, reverse = True)[:elite_size] 
        scores.append(np.average(fitnesses))
        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:            
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            child_gen = []

            rand_a = random.randint(0, dimensions)
            child_gen = [parent1.coordinates[i] if i < rand_a else parent2.coordinates[i] for i in range(dimensions)]
            child = Individual.Individual(child_gen)

            child.gen_len = len(child_gen)
            child.score = ackley.function(individual)           
            offspring.append(child)
            
        # mutation
        mutation_indexes = random.sample(range(len(offspring)), k= MUTATION_INDIVIDUALS)
        for i, index in enumerate(mutation_indexes):         
            # print(f"befor coord {offspring[index].coordinates}")  
            for i, dim in enumerate(offspring[index].coordinates): 
                offspring[index].coordinates[i] *= random.random()
            # print(f"after coord {offspring[index].coordinates}")  
        population = elites + offspring

    # Find the individual with the highest fitness
    best_individual = population[0]
    
    for individual in population:
        individual.score = ackley.function(individual) 
        if best_individual.score < individual.score:
            best_individual = individual

    best_fitness = best_individual.score
    # print_scores_grah(scores)   
    return best_individual.coordinates , best_fitness

def print_scores_grah(scores: list):
    # print(f"the scores are: {scores}")
    max_value_x = len(scores)
    max_value_y = max(scores) + 2
    min_value_x = 0
    min_value_y = min(scores) - 2
    ax = plt.axes()
    plt.suptitle("genetic alorithem scores")
    ax.set(xlim=(min_value_x, max_value_x),
            ylim=(min_value_y, max_value_y),
            xlabel='iterations',
            ylabel='score')
    iterations = [index for index in range(len(scores))]
    plt.plot(iterations, scores)        
    plt.show()
    return
