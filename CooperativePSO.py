# ----------- File Form Lab -----------
import Individual
import Ackley
# ----------- Python Package -----------
import random
import numpy as np
# ----------- Consts Parameters -----------
MAX_TRY = 100


# class SubSwarm:
#     nodes: list
#     particles: list
#     path_solution: list
#     path_score: float
#     cognitive_memory: list
#     social_memory: list
#
#     def __int__(self, nodes):
#         self.nodes = nodes

def cooperative_pso(clusters, start_point):
    best_score = []
    best_solution = []

    for cluster in clusters:
        individuals_index = [ind.index for ind in cluster.individuals]
        # print(f"---------{individuals_index}---------")
        particles = [Particle(cluster.individuals, individuals_index, start_point) for i in range(len(cluster.individuals))]
        global_best = min(particles, key=lambda x: x.score)
        c1, c2, w = 0, 0, 0

        for iteration in range(MAX_TRY):
            c1, c2, w = update_parameters(c1, c2, w, MAX_TRY, iteration)

            for particle in particles:
                particle.update(global_best, c1, c2, w, start_point)

            best_particle = min(particles, key=lambda x: x.personal_best_score)

            if best_particle.personal_best_score < global_best.score:
                best_particle.position = best_particle.personal_best
                best_particle.position_individuals = best_particle.personal_individuals_best
                best_particle.score = best_particle.personal_best_score
                global_best = best_particle

        global_best.position_individuals = [start_point] + global_best.position_individuals + [start_point]
        best_solution.append(global_best.position_individuals)
        best_score.append(global_best.score)

    return best_solution, sum(best_score)


def update_parameters(c1, c2, w, num_iterations, iteration):
    c1 -= 3 * (iteration / num_iterations) + 3.5
    c2 += 3 * (iteration / num_iterations) + 0.5
    w = 0.4 * ((iteration - num_iterations) / num_iterations**2) + 0.4

    return c1, c2, w


class Particle:
    position: list
    nodes: list
    personal_best: list
    personal_best_score: float
    score: float

    def __init__(self, individuals: list, individuals_index: list, start_point: Individual):
        self.position = np.random.permutation(individuals_index)
        self.position_individuals = individuals
        self.position_individuals = self.update_position_individuals()
        self.velocity = np.zeros(len(individuals))
        self.score = self.objective_function(start_point)

        self.personal_best = self.position.copy()
        self.personal_individuals_best = self.position_individuals
        self.personal_best_score = self.score

        return

    def update_position_individuals(self):
        individuals = []
        individuals_index = [ind.index for ind in self.position_individuals]
        for node in self.position:
            node_index_in_individuals = individuals_index.index(node)
            individuals.append(self.position_individuals[node_index_in_individuals])

        return individuals

    def update(self, global_best, c1, c2, w, start_point):

        self.velocity = w * self.velocity + \
                        c1 * random.random() * (self.personal_best - self.position) +\
                        c2 * random.random() * (global_best.position - self.position)

        new_position = self.position + self.velocity
        new_position_sorted = np.argsort(new_position)
        new_position_adjusted = [self.position[new_position_sorted[i]] for i in range(len(new_position_sorted))]
        self.position = np.array(new_position_adjusted)

        self.position_individuals = self.update_position_individuals()
        self.score = self.objective_function(start_point)

        if self.score < self.personal_best_score:
            self.personal_best = self.position.copy()
            self.personal_individuals_best = self.position_individuals.copy()
            self.personal_best_score = self.score

        return

    def objective_function(self, start_point):
        distance = start_point.distance_func(self.position_individuals[0])
        for i in range(len(self.position)-1):
            distance += self.position_individuals[i].distance_func(self.position_individuals[i+1])
        distance += self.position_individuals[len(self.position)-1].distance_func(start_point)

        return distance


# --------------------------------------------------------------------------------------------------------
# ----------- Search Minimum for ackley function -----------
def cooperative_pso_ackley(ackley):
    best_score = 0
    best_solution = None
    num_particle = 1000

    particles = [ParticleAckley(ackley) for i in range(num_particle)]
    global_best = min(particles, key=lambda x: x.score)
    c1, c2, w = 0, 0, 0

    for iteration in range(MAX_TRY):
        c1, c2, w = update_parameters(c1, c2, w, MAX_TRY, iteration)

        for particle in particles:
            particle.update(global_best, c1, c2, w, ackley)

        best_particle = min(particles, key=lambda x: x.personal_best_score)

        if best_particle.personal_best_score < global_best.score:
            best_particle.position = best_particle.personal_best
            best_particle.score = best_particle.personal_best_score
            global_best = best_particle

        best_solution = global_best.position
        best_score = ackley.function_coord(best_solution)

    return best_solution, best_score


class ParticleAckley:
    position: list
    nodes: list
    personal_best: list
    personal_best_score: float
    score: float

    def __init__(self, ackley):
        self.position = np.array([random.uniform(ackley.bounds[0], ackley.bounds[1]) for i in range(ackley.dimensions)])
        self.velocity = np.zeros(ackley.dimensions)
        self.score = ackley.function_coord(self.position)
        self.personal_best = self.position.copy()
        self.personal_best_score = self.score

        return

    def update(self, global_best, c1, c2, w, ackley):

        self.velocity = w * self.velocity + \
                        c1 * random.random() * (self.personal_best - self.position) +\
                        c2 * random.random() * (global_best.position - self.position)

        self.position = self.position + self.velocity
        # self.position = np.clip(self.position, ackley.bounds[0], ackley.bounds[1])
        self.score = ackley.function_coord(self.position)

        if self.score < self.personal_best_score:
            self.personal_best = self.position.copy()
            self.personal_best_score = self.score

        return

