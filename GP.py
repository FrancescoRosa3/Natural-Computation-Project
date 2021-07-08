from typing import Tuple
from Baseline_snakeoil.client import Track
from math import inf
import numpy as np
import matplotlib.pyplot as plt
import json, random, sys, threading, signal
from copy import deepcopy
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET
from threading import Lock, Condition

from numpy.lib.function_base import _percentile_dispatcher, _select_dispatcher

import argparse

import operator
import itertools
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.algorithms import varAnd

import pygraphviz as pgv
import networkx as nx

import dill
import pickle

import heapq

# Load the custom_controller module
import custom_controller_overtake as custom_controller
# define the path were the parameters are defined
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# CONSTANT DEFINITION
NUMBER_SERVERS = 10
BASE_PORT = 3000
MIN_TO_EVALUATE = 3
NUM_RUN_FOR_BEST_EVALUATION = 3

# CONSTANT FOR NORMALIZATION
EXPECTED_NUM_LAPS = 2
MAX_SPEED = 300
FORZA_LENGTH = 5784.10
FORZA_WIDTH = 11.0
WHEEL_LENGHT = 4328.54
WHEEL_WIDTH = 14.0
CG_2_LENGHT = 3185.83
CG_2_WIDTH = 15.0
TRACK_LENGTH = {'forza': FORZA_LENGTH, 'wheel-1': WHEEL_LENGHT, 'g-track-2': CG_2_LENGHT}
UPPER_BOUND_DAMAGE = 1500
UPPER_BOUND_DAMAGE_WITH_ADV = 7000
MAX_OUT_OF_TRACK_TICKS = 1000       # corresponds to 20 sec
OPPONENTS_NUMBER = 8

SAVE_CHECKPOINT = True

# list of track names
track_names = []

# ELEMENTS OF COST FUNCTION
cost_function = {}

# lock
first_lap_failed_agents_lock = Lock()
agents_cnt_lock = Lock()
servers_port_state_lock = Condition(lock=Lock())
POOL = ThreadPool(NUMBER_SERVERS)

# True: server port free, False: server port busy
servers_port_state = [True for i in range(NUMBER_SERVERS)]

# boolean for adversarial
adversarial = True

def clip(param, lb, ub):
    if param < lb:
        return lb
    if param > ub:
        return ub
    return param

# class that defines the problem to solve
class GP_alg():

    # Problem initialization
    def __init__(self, n_pop, mate_prob, mut_prob, n_iter, tournament_size):
        
        #self.controller_variables = controller_variables
        self.n_pop = n_pop
        self.curr_pop = n_pop
        self.mate_prob = mate_prob
        self.mut_prob = mut_prob
        self.n_iter = n_iter-1
        self.tournament_size = tournament_size
        
        #self.pool = pool
        self.toolbox = self.create_toolbox()
        self.agents_cnt = 0
        self.first_lap_failed_agents = 0
        self.fitnesses = []
        self.fitnesses_terms = []
        
        self.best_iter_fit = np.inf
        self.best_iter_fit_terms = None

        self.fitness_correction = False

    # evaluate function
    def _evaluate(self, x):
        
        #def run_simulations(self, x):
        servers_port_state_lock.acquire(blocking=True)
        port_number = 0
        while True:
            try:
                port_number = servers_port_state.index(True)
                servers_port_state[port_number] = False
                print(f"Agent: {self.agents_cnt} got a port.")
                servers_port_state_lock.release()
                break
            except ValueError:
                print(f"Agent: {self.agents_cnt} couldn't get a port.")
                servers_port_state_lock.wait()
                

        # dict where store the fitness for each track
        fitnesses_dict = {}
        # dict where store the fitness component for each track
        fitness_dict_component = {}
        for track in track_names:
            try:
                overtake_func = self.toolbox.compile(expr = x)
                controller = custom_controller.CustomController(overtake_func,
                                                                port=BASE_PORT+port_number+1,
                                                                parameters=self.controller_variables,
                                                                parameters_from_file=False,
                                                                stage=2,
                                                                track=track)
                history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller()
                normalized_ticks = ticks/controller.C.maxSteps

                # compute the number of laps
                num_laps = len(history_lap_time)

                # the car has completed at least the first lap
                # if num_laps > 0 and not race_failed:                
                # compute the average speed
                avg_speed = 0
                max_speed = 0
                min_speed = MAX_SPEED
                for history_key in history_speed.keys():
                    for value in history_speed[history_key]:
                        max_speed = value if value > max_speed else max_speed
                        min_speed = value if value < min_speed else min_speed
                        avg_speed += value
                avg_speed /= ticks
                norm_max_speed = max_speed/MAX_SPEED
                norm_min_speed = min_speed/MAX_SPEED
                #print(f"Num Laps {num_laps} - Average Speed {avg_speed} - Num ticks {ticks}")
                normalized_avg_speed = avg_speed/MAX_SPEED
                distance_raced = history_distance_raced[history_key][-1]
                normalized_distance_raced = distance_raced/(TRACK_LENGTH[track]*EXPECTED_NUM_LAPS)
                # take the damage
                damage = history_damage[history_key][-1]
                global adversarial
                normalized_damage = damage/UPPER_BOUND_DAMAGE if not adversarial else damage/UPPER_BOUND_DAMAGE_WITH_ADV

                # take the car position at the end of the race
                num_laps = 1 if num_laps == 0 else num_laps
                final_car_position = history_car_pos[num_laps][-1]
                final_car_position -= 1
                norm_final_car_position = final_car_position/OPPONENTS_NUMBER

                # take the best position during the race
                best_car_position = 9
                for lap in range(1, num_laps+1):
                    best_car_position = np.min(history_car_pos[lap]) if np.min(history_car_pos[lap]) < best_car_position else best_car_position
                best_car_position -= 1
                norm_best_car_position = best_car_position/OPPONENTS_NUMBER

                # compute out of track ticks and normilize it with respect to the total amount of ticks
                ticks_out_of_track = 0
                for key in history_track_pos.keys():
                    for value in history_track_pos[key]:
                        if abs(value) > 1:
                            ticks_out_of_track += 1
                norm_out_of_track_ticks = ticks_out_of_track/MAX_OUT_OF_TRACK_TICKS                    
                
                # compute the fitness for the current track and store the fitness for the current track
                if not adversarial:
                    fitness = - normalized_avg_speed - norm_max_speed + norm_out_of_track_ticks - normalized_distance_raced# - norm_min_speed 
                    fitness_dict_component[track] = {
                                                        "fitness": fitness, "norm_avg_speed":-normalized_avg_speed, "norm_out_of_track_ticks": norm_out_of_track_ticks,
                                                        "norm_max_speed": norm_max_speed, "normalized_distance_raced": normalized_distance_raced#, "norm_min_speed": norm_min_speed
                                                    }
                else:
                    car_pos_multiplier = 2
                    fitness = (norm_final_car_position * car_pos_multiplier) + norm_best_car_position + norm_out_of_track_ticks  + normalized_damage - normalized_distance_raced 
                    fitness_dict_component[track] = {
                                                        "fitness": fitness, "norm_final_car_position": norm_final_car_position,
                                                        "norm_best_car_position": norm_best_car_position,
                                                        "norm_out_of_track_ticks": norm_out_of_track_ticks, "normalized_damage": normalized_damage,
                                                        "normalized_distance_raced": normalized_distance_raced
                                                    }
                """
                else:
                    if race_failed:
                        print(f"RACE FAILED")
                    else:
                        if not self.fitness_correction:
                            first_lap_failed_agents_lock.acquire(blocking=True)
                            self.first_lap_failed_agents += 1
                            first_lap_failed_agents_lock.release()
                    fitness = 10 
                #return fitness
                """
                
                if num_laps == 0:
                    if not self.fitness_correction:
                        first_lap_failed_agents_lock.acquire(blocking=True)
                        self.first_lap_failed_agents += 1
                        first_lap_failed_agents_lock.release()
                
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                fitness = 20

            fitnesses_dict[track] = fitness
            
        # compute the average performance over all the tested tracks
        total_fitness = 0
        num_track = 0
        for fitness_on_track in fitnesses_dict.keys():
            total_fitness += fitnesses_dict[fitness_on_track]
            num_track += 1
        total_fitness /= num_track

        if total_fitness < self.best_iter_fit:
            self.best_iter_fit = total_fitness
            self.best_iter_fit_terms = fitness_dict_component
        #self.fitnesses.append(total_fitness)
        #self.fitnesses_terms.append(fitness_dict_component)


        if not self.fitness_correction:
            agents_cnt_lock.acquire(blocking=True)
            self.agents_cnt += 1
            print(f"Agent runned {self.agents_cnt}",end="\r")
            agents_cnt_lock.release()
        
        servers_port_state_lock.acquire(blocking=True)
        servers_port_state[port_number] = True
        servers_port_state_lock.notify_all()
        servers_port_state_lock.release()
        
        if not self.fitness_correction:
            return total_fitness,
        return total_fitness, fitness_dict_component
        #return run_simulations(self, x)
       
    def create_toolbox(self):
        # defined a new primitive set for strongly typed GP
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 7), float, "ARG")

        # boolean operators
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.not_, [bool], bool)

        # floating point operators
        # Define a protected division function
        def protectedDiv(left, right):
            try:
                return left / right
            except ZeroDivisionError:
                return 1

        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protectedDiv, [float, float], float)

        # logic operators
        # Define a new if-then-else function
        def if_then_else(input, output1, output2):
            if input:
                return output1
            else:
                return output2

        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        # pset.addPrimitive(operator.eq, [float, float], bool)
        pset.addPrimitive(if_then_else, [bool, float, float], float)

        # terminals
        pset.addEphemeralConstant("vel", lambda: random.randint(0, 300), int)
        pset.addEphemeralConstant("dist", lambda: random.randint(5, 20), int)
        pset.addEphemeralConstant("randpi", lambda: random.uniform(-np.pi, np.pi), float)

        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)

        #input
        pset.renameArguments(ARG0='sti')
        pset.renameArguments(ARG1='tp')
        pset.renameArguments(ARG2='a')
        pset.renameArguments(ARG3='ttp')
        pset.renameArguments(ARG4='sx')
        pset.renameArguments(ARG5='op_left')
        pset.renameArguments(ARG6='op_right')

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("evaluate", self._evaluate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        return toolbox

    def parallelize(self, pool):
        self.toolbox.register("map", pool.map)

    def run_problem(self):
        pop = self.toolbox.population(n=self.n_pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = self.eaSimple(pop, self.toolbox, self.mate_prob, self.mut_prob, self.n_iter, stats, halloffame=hof, verbose=True)

        return pop, stats, hof, logbook
    
    def add_parameters(self, parameters):
        self.controller_variables = parameters
    
    def on_iter_end(self):
        
        # print iteration stats
        print(F"# OF AGENTS THAT FAILED FIRST LAP: {self.first_lap_failed_agents}/{self.curr_pop*len(track_names)}")
        print(f"BEST fitness: {self.best_iter_fit}")
        print(f"TERMS:\n{self.best_iter_fit_terms}")
        
        # reset counters
        self.best_iter_fit = np.inf
        self.agents_cnt = 0
        self.first_lap_failed_agents = 0

    def eaSimple(self, population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
            
        def update_top_k_ind_fitness(population, k=3):
        # update the fitness of the k best individuals of the population repeating
        # the race three times for each of them
            
            self.fitness_correction = True
            # get the k best individuals
            inds_list = []
            for i, ind in enumerate(population):
                inds_list.append(ind.fitness.values[0])
            inds_list = np.array(inds_list)
            print(np.sort(inds_list))
            best_k_fits_index = np.argpartition(inds_list, k)[:k]

            best_k_inds = []
            for i in best_k_fits_index:
                best_k_inds.append(population[i])

            # update their fitness
            race_to_repeat = 3
            inds_fitnesses = []
            inds_terms = []
            for i in range(race_to_repeat):
                evaluate_return = toolbox.map(toolbox.evaluate, best_k_inds)
                inds_fitnesses_temp = []
                inds_terms_temp = []
                for fitnesses, terms in evaluate_return:
                    inds_fitnesses_temp.append(fitnesses)
                    inds_terms_temp.append(terms)
                inds_fitnesses.append(inds_fitnesses_temp)
                inds_terms.append(inds_terms_temp)
                
            inds_fitnesses = np.array(inds_fitnesses)
            inds_fitnesses = np.average(inds_fitnesses, axis=0)

            for i in range(len(best_k_inds)):
                best_k_inds[i].fitness.values = (inds_fitnesses[i], )

            # get best individual
            print(np.sort(inds_fitnesses))
            best_ind_index = np.argmin(inds_fitnesses)
            best_ind = best_k_inds[best_ind_index]

            # print iteration stats
            print(F"# OF AGENTS THAT FAILED FIRST LAP: {self.first_lap_failed_agents}/{self.curr_pop*len(track_names)}")
            print(f"BEST fitness: {best_ind.fitness.values[0]}")
            for race in inds_terms:
                print(f"TERMS:\n{race[best_ind_index]}")
            
            # reset counters
            self.best_iter_fit = np.inf
            self.agents_cnt = 0
            self.first_lap_failed_agents = 0
            
            self.fitness_correction = False


        global adversarial
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        self.curr_pop = len(invalid_ind)
        print(F"INVALID #: {len(invalid_ind)}")
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if adversarial:
            update_top_k_ind_fitness(population)
        else:
            self.on_iter_end()

        if halloffame is not None:
            halloffame.update(population)
        

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        save_results(population, halloffame, logbook, 0)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.curr_pop = len(invalid_ind)
            print(F"INVALID #: {len(invalid_ind)}")
            
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            print(f"COMPUTED FITNESS OF GEN: {gen}")
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            if adversarial:
                print(f"UPDATING FITNESS OF GEN: {gen}")
                update_top_k_ind_fitness(offspring)
                print(f"UPDATED FITNESS OF GEN: {gen}")
            else:
                self.on_iter_end()

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
                
            save_results(population, halloffame, logbook, gen)

        return population, logbook


def take_track_names(args):
    track_names = []

    path_name = dir_path + "/configuration_file/" + args.configuration_file + f"_{1}.xml"
    
    configuration_file = ET.parse(path_name)
    root = configuration_file.getroot()

    for child in root:
        if child.attrib['name'] == 'Tracks':
            for section in child.findall('section'):
                for sub_section in section.findall('attstr'):
                    if sub_section.attrib['name'] == 'name':
                        track_names.append(sub_section.attrib['val'])
    return track_names 

def create_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

"""
def save_checkpoint(algorithm, iter):
    global np_seed, de_seed, n_pop, max_gens, n_vars, cr, f, PARAMETERS_STRING
    checkpoint_folder = results_folder + "/Checkpoints/"+ PARAMETERS_STRING + "/"
    create_dir(checkpoint_folder)
    checkpoint_file_name = checkpoint_folder + f"iter-{iter}.npy"
    try:
        with open(checkpoint_file_name, 'wb') as file:
            np.save(file, algorithm)
        file.close()
        print(f"iteration {iter} checkpoint created")
        return checkpoint_file_name
    except:
        print(f"iteration {iter} checkpoint creation failed")
        return None

def load_checkpoint(checkpoint_file_name):
    try:
        with open(checkpoint_file_name, 'rb') as file:
            checkpoint, = np.load(file, allow_pickle=True).flatten()
            last_iteration = checkpoint_file_name.split("-")[-1].split(".")[0]
            last_iteration = int(last_iteration)
        file.close()
        print(f"iteration {last_iteration} checkpoint restored")
        return checkpoint, last_iteration
    except:
        print(f"iteration {last_iteration} checkpoint could not be restored")
        return None, None


def get_configuration(path):
    conf_split_underscore = path.split("_")
    version = conf_split_underscore[-1].split(".")[0]
    print(f"version: {version}")
    pfile= open(f"{dir_path}\{path}",'r')
    print(f"condition_path: {dir_path}\{path}")
    parameters_to_change = json.load(pfile)
    if conf_split_underscore[-4] == 'no':
        global adversarial
        adversarial = False
        print("Adversarial {adversarial}")
    return parameters_to_change, version

def save_results(result_params):
    global parameters_to_change, parameters, results_folder
    print("Saving the best result on file.....")
    i = 0
    for key in parameters_to_change.keys():
        # change the value of contreller_variables
        # if the given variable is under evolution
        if parameters_to_change[key][0] == 1:
            # this parameter is under evolution
            parameters[key] = result_params[i]
            i += 1
    
    create_dir(results_folder)
    file_name = results_folder  + "/" + PARAMETERS_STRING + ".xml"
    
    with open(file_name, 'w') as outfile:
        json.dump(parameters, outfile)
"""

def save_results(pop, hof, logbook, iter):

    cp = dict(population = pop, generation=iter, halloffame=hof,
                logbook=logbook, rndstate=random.getstate())

    global results_folder
    create_dir(results_folder)
    file_name = results_folder  + "/" + PARAMETERS_STRING

    nodes, edges, labels = gp.graph(hof.items[0])
    
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

    g.draw(file_name + f"_tree_{iter}.pdf")

    file_name += f"_{iter}.pkl"
    with open(file_name, "wb") as cp_file:
        dill.dump(cp, cp_file)

def save_hof(hof, iter):

    global results_folder
    create_dir(results_folder)
    file_name = results_folder  + "/" + PARAMETERS_STRING

    nodes, edges, labels = gp.graph(hof.items[0])

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

    g.draw(file_name + f"_tree_{iter}.pdf")
    
    file_name += f"_hof_{iter}.pkl"
    with open(file_name, "wb") as cp_file:
        dill.dump(hof, cp_file)
    

np_seed = 32
# set the np seed
np.random.seed(np_seed)

# Pymoo Differential Evolution
print('Pymoo Differential Evolution')

# population size
n_pop = 50
# mate prpbability
mate_prob = 0.5
# mutation prpbability
mut_prob = 0.5
# maximum number of generations
n_iter = 20
# number of individuals in each tournament
tournament_size = int(n_pop/10)


PARAMETERS_STRING = f"{np_seed}_{n_pop}_{mate_prob}_{mut_prob}_{n_iter}_{tournament_size}"

# define the problem
algorithm = GP_alg(n_pop, mate_prob, mut_prob, n_iter, tournament_size)
    
if __name__ == "__main__":
    ####################### SETUP ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', '-cp', help="checkpoint file containing the starting population for the algorithm", type= str,
                        default= "None")
    parser.add_argument('--configuration_file', '-conf', help="name of the configuration file to use, without extension or port number", type= str,
                    default= "quickrace_forza_no_adv")
    parser.add_argument('--controller_params', '-ctrlpar', help="initial controller parameters", type= str,
                    default= "Baseline_snakeoil\default_parameters")
    args = parser.parse_args()

    # get opponents presence from configuration file
    if args.configuration_file.split("_")[-2] == 'no':
        adversarial = False
        print(f"Adversarial {adversarial}")
    
    # load default parameters
    args.controller_params = '\\' + args.controller_params
    pfile= open(dir_path + args.controller_params,'r') 
    parameters = json.load(pfile)

    track_names = take_track_names(args)

    tracks_folder = args.configuration_file
    results_folder = dir_path+"/Results_GP/"+ tracks_folder
    create_dir(results_folder)

    if args.checkpoint_file == "None":

        pool = multiprocessing.dummy.Pool(NUMBER_SERVERS)

        algorithm.add_parameters(parameters)
        algorithm.parallelize(pool)

        pop, stats, hof, logbook = algorithm.run_problem()

        best_individual_so_far = hof.items[0]
        best_fitness_so_far = hof.items[0].fitness.values[0]

        print("hof phenotype: ", best_individual_so_far)
        print("hof fitness: ", best_fitness_so_far, best_fitness_so_far/n_pop)

        nodes, edges, labels = gp.graph(hof.items[0])

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        save_results(pop, hof, logbook, n_iter)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()
    
    else:
        with open(args.checkpoint_file, "rb") as cp_file:
            cp = dill.load(cp_file)
        #population = cp["population"]
        #start_gen = cp["generation"]
        #halloffame = cp["halloffame"]
        #logbook = cp["logbook"]
        #print(cp)
        """
        random.setstate(cp["rndstate"])
        print(population)
        print(start_gen)
        print(halloffame)
        print(logbook)
        """
        
        """
        # when only hof is saved
        toolbox = algorithm.toolbox
        #hof = cp["halloffame"]
        overtake_func = cp
        overtake_func_compiled = toolbox.compile(expr = overtake_func.items[0])
        controller = custom_controller.CustomController(overtake_func_compiled,
                                                                    port=BASE_PORT+1,
                                                                    parameters=parameters,
                                                                    parameters_from_file=False,
                                                                    stage=2,
                                                                    track="forza")
        for i in range(10):
            history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller()
            print(history_lap_time)
                
        """

        #"""
        # when everything is saved
        hof = cp["halloffame"]
        toolbox = algorithm.toolbox
        #hof = cp["halloffame"]
        overtake_func = hof
        overtake_func_compiled = toolbox.compile(expr = overtake_func.items[0])
        controller = custom_controller.CustomController(overtake_func_compiled,
                                                                    port=BASE_PORT+1,
                                                                    parameters=parameters,
                                                                    parameters_from_file=False,
                                                                    stage=2,
                                                                    track="wheel-1")
        for i in range(10):
            history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller()
            print(history_lap_time)
                
        #"""

        