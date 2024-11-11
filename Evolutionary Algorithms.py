# 

import random
import copy
import numpy as np
from deap import base
from deap import creator
from deap import tools
from matplotlib import pyplot as plt


#The following function calculates payoff. Easier to do it from one player, say the first (parameter) individual's
#perspective. The last argument "game" is for the "game payoff matrix." 
#Suppse you use 0 for move "V" and 1 for "A". Then the move determines the row of first matrix 
#(Table 1 on page 1 of Assignment Specs) from the "blue" player's perspective. 

def payoff_to_ind1(individual1, individual2, individual3, game):

#Your code replaces "pass". 

    payoff = 0
    if individual1[-1] == 1:
        if individual2[-1] == 1:
            if individual3[-1] == 1:
                payoff = game[0][0][0]
            else:
                payoff = game[0][1][0]
        else:
            if individual3[-1] == 1:
                payoff = game[0][2][0]
            else:
                payoff = game[0][3][0]
    else:
        if individual2[-1] == 1:
            if individual3[-1] == 1:
                payoff = game[1][0][0]
            else:
                payoff = game[1][1][0]
        else:
            if individual3[-1] == 1:
                payoff = game[1][2][0]
            else:
                payoff = game[1][3][0]
    return payoff 

# implements from individual1's perspective
# use the stored "history bits" -- the last two moves by the three players -- to do lookup.
# use the default moves for the first two rounds.

def move_by_ind1(individual1, individual2, individual3, round):

    move = 0
    if round == 1:
        move =  individual1[64]
    elif round == 2:
        move =  individual1[65]
    else:
        aux = str(individual1[-2]) + str(individual1[-1]) + str(individual2[-2]) + str(individual2[-1]) + str(individual3[-2]) + str(individual3[-1])
        aux = int(aux, 2)
        move = individual1[aux]
    return move
    

#Once the moves have been made, the history bits are to be updated.
# This function does that. 

def process_move(individual, move, memory_depth):

#Your code replaces "pass". 
    if memory_depth == 2:
        individual[66] = individual[67]
        individual[67] = move
    else:
        individual = individual

# Play n_rounds number of rounds. It has been fixed at 5 below (in the "main"). 
# Returns the score of all players after the "match" consisting of num_rounds. 
# Again, be careful of order. 

def eval_function(individual1, individual2, individual3, m_depth, n_rounds):

    matrix_3DV = [[[8,8,8],[6,6,7],[6,7,6],[5,9,9]],[[7,6,6],[9,5,9],[9,9,5],[0,0,0]]]

    score1 = 0
    score2 = 0
    score3 = 0
    
    for i in range(n_rounds):
        move1 = move_by_ind1(individual1, individual2, individual3, i)
        move2 = move_by_ind1(individual2, individual3, individual1, i)
        move3 = move_by_ind1(individual3, individual1, individual2, i)

        process_move(individual1, move1, m_depth)
        process_move(individual2, move2, m_depth)
        process_move(individual3, move3, m_depth)

        score1 += payoff_to_ind1(individual1, individual2, individual3, matrix_3DV)
        score2 += payoff_to_ind1(individual2, individual3, individual1, matrix_3DV)
        score3 += payoff_to_ind1(individual3, individual1, individual2, matrix_3DV)

    return score1, score2, score3

##The following function is. You can use unscaled (raw) fitness. 
###Raw fitness values given by eval_function tend to get stuck. So the fitness values are scaled (see Haider's paper).
## Attempt ONLY AFTER you have finished the rest, for some bonus points!
##You can earn full mark without implementing it.

def calculate_scaled_fitnesses(pop, raw_fitness):
  
    c = 2
    a = (c-1)*(np.mean(raw_fitness)/(np.max(raw_fitness)-(np.mean(raw_fitness))))
    b = np.mean(raw_fitness)*((np.mean(raw_fitness)-(c*np.mean(raw_fitness)))/(np.max(raw_fitness)-np.mean(raw_fitness)))
    raw_fitness = (a*raw_fitness) + b


#The eval_function computes the raw_fitness of an individual in match
#The following function will compute the accumulated fitness of individuals across all the matches in a tournament
#The accumulated fitness will be the sum of the scores of an individual after a tournament. 

def fitness_tournament(pop, raw_fitness, rounds):

    for i in range(len(raw_fitness)):
        for j in range(len(raw_fitness)):
            for k in range(j+1):
                score1, _, _ = eval_function(pop[i], pop[j], pop[k], 2, rounds)    
                raw_fitness[i] += score1

#############################
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute (gene) generator -- 0 or 1, randomly chosen with 50% probability

toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers -- define 'individual' to be a bit array of appropriate length.

n_players = 3
m_depth = 2
strategies = 64 # the number of strategy bits
chrom_len = 68 # size of an individual

# define an individual
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, chrom_len)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ----------
# Operator registration
# ----------

# register the goal / fitness function
toolbox.register("evaluate", eval_function)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting parents:
# each individual of the current generation is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation. We use "selTournament" selection.

toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    game_n_rounds = 10 # Number of rounds in a match
    random.seed(64)
    pop_size = 5    #Populations size. Set to 5, you may start with 3 (no tournament). Then increase it. 
    NO_ITERATION = 100 #You can start small and then increase it
    pop = toolbox.population(pop_size)

    # Crossover and Mutation rates
    CXPB, MUTPB = 0.95, 0.01

    print("Start of evolution")

    # Evaluate the entire population
    # Fitness of each player is the sum of each player's payoff after n_rounds have been played
    raw_fitness = np.zeros(pop_size)
    fitness_tournament(pop, raw_fitness, game_n_rounds)
    
    #raw_fitness should be an array of size pop_size. The fitness values (=total score after a tournament)
    #is stored for each individual. Make sure the indexes match! For example, the individual at index 0 
    #in population should have its fitness scored at 0 in raw_fitness. 

    # Variable keeping track of the number of generations
    g = 0
    best_original_fit = 0
    # Begin the evolution
    while g < NO_ITERATION:
        # A new generation
        g = g + 1
        print("-- Generation %g --" % g)

        # Select individuals to be "parents"
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop = copy.deepcopy(offspring)

        # Reset raw_fitnesses
        raw_fitness = np.zeros(pop_size)
        fitness_tournament(pop, raw_fitness, game_n_rounds)
        
        #Use only if you are using scaled fitness. 
        #calculate_scaled_fitnesses(pop, raw_fitness)
        
        for ind, fit in zip(pop, raw_fitness):
            ind.fitness.values = fit,


        best_original_fit = np.round(np.max(raw_fitness), 3)
        average_fitness = np.round(np.mean(raw_fitness), 3)

        print("  Best %s" % best_original_fit)
        print("  Avg %s" % average_fitness)

    print("-- End of (successful) evolution --")

    return 


if __name__ == "__main__":
    main()
