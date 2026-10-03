from search_problem import AirportProblem
from scipy.optimize import minimize
import pygad
import numpy as np
import matplotlib.pyplot as plt


# instantiate the search problem
problem = AirportProblem(n_cities=30, n_airports=3)


# TASK 1: solve for best airport locations using the gradient-descent (CG) approach
# YOUR CODE HERE

x0 = np.random.rand(2 * problem.n_airports)
cg_result = minimize(problem.sum_city_distances, x0, method='CG')

print(cg_result) 
print("Airport locations: ", cg_result.x)
print("Sum of distances: ", cg_result.fun)



# TASK 2: solve for best airport locations using a genetic algorithm approach
# YOUR CODE HERE

def fitness_func(ga_instance, solution, solution_idx):
    '''
    Implement your fitness function here.
    :param ga_instance: a reference to the GA object - this param is just to comply with PyGad's required function signature; you wont need it
    :param solution: the candidate solution to measure fitness for: an array of locations [x1, y1, x2, y2...]
    :param solution_idx: the index of this solution within the bigger population. You won't need to use this either.
    :return: the solution's fitness score (Remember that GAs *maximize* fitness, but we want to *minimize* distance)
    '''
    return -problem.sum_city_distances(solution) # minimizing the distance

def xover_func(parents, offspring_size, ga_instance):
    '''
    Implement your crossover function here.
    A good crossover should combine information from two (or more) parents in some way.
    :param parents: an array containing all the parents to use in this round of crossovers (the number of parents selected is controlled by the GA params)
    :param offspring_size: the number of new (crossed-over) solutions this function should produce
    :param ga_instance: a reference to the GA object - this param is just to comply with PyGad's required function signature; you wont need it
    :return: array of new solutions
    '''

    offspring = [] #empty array for the children 
    for k in range(offspring_size[0]):
        p1 = parents[np.random.randint(parents.shape[0])]
        p2 = parents[np.random.randint(parents.shape[0])]

        child = p1.copy() #let child be a copy of its parent

        for a in range(problem.n_airports):
            # if its below 0.5 take this airport from parent 2, if it's above 0.5 then keep the parent 1's airport
            if np.random.rand() < 0.5:
                child[2 * a] = p2[2 * a] # x coordinate
                child[2 * a + 1] = p2[2 * a + 1] # y coordinate
        offspring.append(child)
    return np.array(offspring)

# instantiate the PyGad GA object
# for more info on GA params: https://pygad.readthedocs.io/en/latest/pygad.html#init
# for defining your own selection, mutation, or crossover operations:
#    https://pygad.readthedocs.io/en/latest/utils.html#user-defined-crossover-mutation-and-parent-selection-operators
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=20,
                       fitness_func=fitness_func,
                       sol_per_pop=100,
                       num_genes=problem.n_airports * 2, # length of a solution vector (x-y coords for each airport)
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="tournament",
                       crossover_type=xover_func,
                       mutation_type='random',
                       mutation_by_replacement=True,
                       random_mutation_min_val=0,
                       random_mutation_max_val=1,
                       mutation_percent_genes=20)


# TASK 3
# plot the cities and the solutions found
# YOUR CODE HERE

def plot_cities(problem, solution, title):
    plt.figure(figsize=(8, 8))
    
    plt.scatter(
            problem.city_locations[:, 0],
            problem.city_locations[:, 1],
            c='blue',
            label='Cities'
    )
    
    plt.scatter(
        solution[0::2],
        solution[1::2],
        c='red',
        label='Airports',
        marker='X',
        s=100
    )
        
    plt.title(title)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    plt.legend()
    plt.grid()
    plt.show()


# run and get results
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution_fitness)
plot_cities(problem, solution, "Genetic Algorithm Solution")

# TASK 4: 
# What insights did you gain into genetic algorithms during this lab?
'''
The genetic algorithm shows that it covers a wide range of solutions and it takes a while to search through.
Choosing the best fit based on the which parent solutions are selected and how the crossover is done is important to get a good solution.
'''


# GAs are considered to be particularly powerful (if slow) search algorithms.
# Did you find they significantly outperformed the CG method in this problem? Why do you think that is?

'''
Through searching different solutions in a wider search area with different contexts rather than a fixed array of values, it is
believed that is where there is a better chancve of finding a better solution.
'''

