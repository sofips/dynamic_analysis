import numpy as np
from dgamod import *

# from zhang_actions_mod import *
import csv
import pygad
import sys
import time
import os
import configparser


# get parameters from config file
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, str(sys.argv[1]))
print(initfile)
config = configparser.ConfigParser()
config.read(initfile)


# system parameters
n = config.getint("system_parameters", "n")
dt = config.getfloat("system_parameters", "dt")
b = config.getfloat("system_parameters", "b")
speed_fraction = config.getfloat("system_parameters", "speed_fraction") # fraction of qsl speed if loc based fitness
max_optimization_time = config.getint("system_parameters", "max_optimization_time")

# aca genero las acciones
acciones = one_field_actions(b, n)  
props = gen_props(acciones, n, b, dt)


# genetic algorithm parameters
num_generations = config.getint("ga_initialization", "num_generations")
num_genes = config.getint("ga_initialization", "num_genes")
sol_per_pop = config.getint("ga_initialization", "sol_per_pop")
fidelity_tolerance = config.getfloat("ga_initialization", "fidelity_tolerance")
saturation = config.getint("ga_initialization", "saturation")
reward_decay = config.getfloat("ga_initialization", "reward_decay")


# crossover and parent selection

num_parents_mating = config.getint("parent_selection", "num_parents_mating")
parent_selection_type = config.get("parent_selection", "parent_selection_type")
keep_elitism = config.getint("parent_selection", "keep_elitism")
crossover_type = config.get("crossover", "crossover_type")
crossover_probability = config.getfloat("crossover", "crossover_probability")

# mutation
mutation_probability = config.getfloat("mutation", "mutation_probability")
mutation_num_genes = config.getint("mutation", "mutation_num_genes")


gene_space = np.arange(0, n, 1) # un gen por sitio
gene_type = int

stop_criteria = ["saturate_" + str(saturation)]  # , 'reach_'+str(fidelity_tolerance)]

population_histograms = config.getboolean("saving","population_histograms")
dirname = config.get("saving", "directory")
n_samples = config.getint("saving", "n_samples")

filename = dirname + "/nvsmaxfid.dat"

# call construction functions
on_generation = generation_func_constructor(
    generation_func, [props, fidelity_tolerance, dirname, population_histograms]
)

fidelity_args = [props,speed_fraction, max_optimization_time]#,fidelity_tolerance,reward_decay]
fitness_func = fitness_func_constructor(localization_based, fidelity_args)
mutation_type = "swap"


with open(filename, "a") as f:
    for i in range(n_samples):
        writer = csv.writer(f, delimiter=" ")

        solutions_fname = "{}/act_sequence_n{}_sample{}.dat".format(dirname, n, i)
        # fitness_history_fname = dirname + '/fitness_history_sample'+ str(i) + '.dat'

        t1 = time.time()

        initial_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            parent_selection_type=parent_selection_type,
            keep_elitism=keep_elitism,
            gene_space=gene_space,
            gene_type=gene_type,
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            mutation_type=mutation_type,
            on_generation=on_generation,
            mutation_num_genes=mutation_num_genes,
            stop_criteria=stop_criteria,
            save_solutions=False
        )

        initial_instance.run()

        t2 = time.time()
        trun = t2 - t1

        maxg = initial_instance.generations_completed

        solution, solution_fitness, solution_idx = initial_instance.best_solution()

        evolution = time_evolution(solution, props, n, graph=False, filename=False)
        time_max_fidelity = np.argmax(evolution) * dt

        row = [
            n,
            i,
            format(fidelity(solution, props)),
            "{:.8f}".format(time_max_fidelity),
            maxg,
            "{:.8f}".format(trun),
        ]
        writer.writerow(row)
        actions_to_file(solution, solutions_fname, "w")
