# inputs_text_problem = "Select a problem:\nString = 0\nN-Queens = 1\nBinPacking = 2\nCartesian function = 3\n"
# inputs_text_num_genes = "enter the size of the gen:\nFor String enter 13\nFor N-Queens enter 8\nFor BinPacking enter 0\nFor Cartesian enter 2\n"
# inputs_text_fitness_function = "Enter the fitness function:\noriginal = 0 \nString bulls eye = 1\nBitwise bulls eye = 2\n"
# inputs_text_mutation_selection = "Enter the mutation:\nBinPacking None = 0 \nString mutation= 1\nN-Queens invertion shuffle = 2\nN-Queens Just shuffle = 3\nCartesian mutation = 4\n"
# inputs_text_crossover_operator = "Enter the number of crossover operator:\nNone = 0 \nSingel = 1\nTwo = 2\nUniform = 3\nN-Queens PMX = 4\nN-Queens CX = 5\nBIN_PACKING = 6\nCartesian = 7\n"
# inputs_text_mutation_control_selection = "Enter mutation control method:\nNONE = 0 \nConst mutation= 1\nDecrease linearly = 2\nNon Linear logistic decay = 3\n"
# inputs_text_parent_selection = "Enter the parent selection oprator:\nNone = 0 \nRWS = 1 \nSUS = 2\nTOURNAMENT RANKING = 3\n"
# inputs_text_niche_algorithm = "Enter the niche algorithm:\n (Share Fitness = 0) \nClustering = 1 \nCrowding = 2\n"
# inputs_text_age_factor = "Enter the age factor:\n"
# inputs_text_viability_fuc_num = "Enter the viability func num:\nNone = 0\nf1 = 1\nf2 = 2\n"
# inputs_text_algorithm = "Enter the algorithm:\nTabu search = 0\nACO = 1\nSimulated Annealing = 2\nGA = 3\nCooperative PSO = 4\n"


class Data:

    def __init__(self, setting_vector=None):
        if setting_vector:
            self.init_with_settings(setting_vector)
        else:
            self.init_with_user_input()

        self._init_consts()
        return

    def init_with_user_input(self):
        # self.problem = int(input(inputs_text_problem))
        # self.num_genes = int(input(inputs_text_num_genes))
        # self.fitness_function = int(input(inputs_text_fitness_function))
        # self.mutation_selection = int(input(inputs_text_mutation_selection))
        # self.cross_operator = int(input(inputs_text_crossover_operator))
        # self.mutation_control_selection = int(input(inputs_text_mutation_control_selection))
        # self.parent_selection = int(input(inputs_text_parent_selection))
        # self.niche_algorithm = int(input(inputs_text_niche_algorithm))
        # self.age_factor = float(input(inputs_text_age_factor))
        # self.viability_fuc_num = int(input(inputs_text_viability_fuc_num))
        # self.algorithm = inputs_text_algorithm
        return

    def init_with_settings(self, setting_vector):
        # self.problem = setting_vector[0]
        # self.num_genes = setting_vector[1]
        # self.fitness_function = setting_vector[2]
        # self.mutation_selection = setting_vector[3]
        # self.cross_operator = setting_vector[4]
        # self.mutation_control_selection = setting_vector[5]
        # self.parent_selection = setting_vector[6]
        # self.niche_algorithm = setting_vector[7]
        # self.age_factor = setting_vector[8]
        # self.viability_fuc_num = setting_vector[9]
        # self.algorithm = inputs_text_algorithm[10]
        return

    def _init_consts(self):
        # self.pop_size = 100
        # self.max_generations = 100
        # self.max_age = 101
        return
