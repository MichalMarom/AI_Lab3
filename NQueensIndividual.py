# ----------- File Form Lab -----------
import math
import Individual
import Data
# ----------- Python Package -----------
import numpy as np
import random
# ----------- Consts Name  -----------
ORIGINAL_FIT = 0


class NQueensIndividual:
    gen: list
    start_point: Individual
    end_point: Individual
    score: float

    def __init__(self, 
                 data: Data, 
                 individuals: list, 
                 start_point, 
                 end_point):
        # self.gen = random.sample(range(1, data.num_genes+1), data.num_genes)
        self.gen = individuals
        random.shuffle(self.gen)
        self.start_point = start_point
        self.end_point = end_point

        self.gen_len = len(individuals)
        # self.age = 0
        self.score = 0
        # self.fitness_function = data.fitness_function
        self.update_score(data)

    def update_score(self, data: Data):
        self.score = self.original_fitness(data)
        return 
    
    def original_fitness(self, data: Data):
        total_distance = 0

        self.gen.insert(0 ,self.start_point)
        self.gen.append(self.end_point)

        for i in range(len(self.gen)-1):
            x1, y1 = self.gen[i].coordinates
            x2, y2 = self.gen[i+1].coordinates
            total_distance -= math.sqrt((x2-x1)**2 + (y2-y1)**2)

        self.gen.remove(self.gen[0])
        self.gen.remove(self.gen[-1])
        return total_distance

    def mutation(self, data: Data):
        self.just_shuffle()
        return

    def just_shuffle(self):
        random.shuffle(self.gen)

        return
