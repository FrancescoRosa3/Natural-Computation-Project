from typing import Tuple
from Baseline_snakeoil.client import Track
from math import inf
import numpy as np
import matplotlib.pyplot as plt
import json, random, sys, threading, signal
from copy import deepcopy
import time
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET
from threading import Lock, Condition

from numpy.lib.function_base import _select_dispatcher

from pymoo.algorithms.so_de import DE
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_visualization, get_termination
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

from pymoo.model.problem import Problem

import argparse

# Load the custom_controller module
import custom_controller
# define the path were the parameters are defined
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# CONSTANT DEFINITION
NUMBER_SERVERS = 10
BASE_PORT = 3000
PERCENTAGE_OF_VARIATION = 40
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
agents_cnt_lock = Lock()
POOL = ThreadPool(NUMBER_SERVERS)

servers_port_state_lock = Condition(lock=Lock())
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
class TorcsProblem(Problem):

    # Problem initialization
    def __init__(self, variables_to_change, controller_variables, lb, ub):
        
        """super().__init__(n_var=lb.shape[0], n_obj=1, n_constr=lb.shape[0], 
                        xl=np.array([-100000 for i in range(lb.shape[0])]), xu=np.array([100000 for i in range(lb.shape[0])]))#,xl = lb, xu = ub)
        """
        super().__init__(n_var=lb.shape[0], n_obj=3, n_constr=0, 
                        xl=np.array([-100000 for i in range(lb.shape[0])]), xu=np.array([100000 for i in range(lb.shape[0])]))#,xl = lb, xu = ub)
        
        self.variable_to_change = variables_to_change
        self.controller_variables = controller_variables
        self.lb = lb
        self.ub = ub

        global n_pop
        self.agents_cnt = 0
        self.fitness_terms = [None for i in range(n_pop)]
           
    # evaluate function
    def _evaluate(self, X, out, *args, **kwargs):
        
        # restart evaluated agents counter
        #agents_cnt_lock.acquire(blocking=True)
        self.agents_cnt = 0
        #agents_cnt_lock.release()

        def run_simulations(x, agent_indx, variable_to_change, controller_variables):
            servers_port_state_lock.acquire(blocking=True)
            port_number = 0
            while True:
                try:
                    port_number = servers_port_state.index(True)
                    servers_port_state[port_number] = False
                    servers_port_state_lock.release()
                    #print(f"Agent {agent_indx}- Found Free port {port_number}")
                    break
                except ValueError:
                    #print(f"Agent {agent_indx} wait....")
                    servers_port_state_lock.wait()
                    #print(f"Agent {agent_indx} waked up")
            

            i = 0
            for key in variable_to_change.keys():
                # change the value of contreller_variables
                # if the given variable is under evolution
                if variable_to_change[key][0] == 1:
                    # this parameter is under evolution
                    #print(f"key: {key} - starting value: {controller_variables[key]:.2f} - modified value: {x[agent_indx][i]}")
                    controller_variables[key] = x[i]
                    i += 1

            # dict where store the fitness for each track
            fitnesses_dict = {}
            # dict where store the fitness component for each track
            fitness_dict_component = {}
            for track in track_names:
                try:
                    #print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                    controller = custom_controller.CustomController(port=BASE_PORT+port_number+1,
                                                                    parameters=controller_variables, 
                                                                    parameters_from_file=False,
                                                                    stage=2,
                                                                    track=track)
                    
                    history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks, race_failed = controller.run_controller()
                    
                    normalized_ticks = ticks/controller.C.maxSteps

                    # compute the number of laps
                    num_laps = len(history_lap_time)

                    # the car has completed at least the first lap
                    if num_laps > 0 and not race_failed:
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
                        car_position = history_car_pos[history_key][-1]
                        car_position -= 1
                        norm_car_position = car_position/OPPONENTS_NUMBER

                        # compute the average from the center line
                        """
                        average_track_pos = 0
                        steps = 0
                        for key in history_track_pos.keys():
                            for value in history_track_pos[key]:
                                steps += 1
                                if abs(value) > 1:
                                    average_track_pos += (abs(value) - 1)
                        average_track_pos /= steps
                        """

                        # compute out of track ticks and normilize it with respect to the total amount of ticks
                        ticks_out_of_track = 0
                        for key in history_track_pos.keys():
                            for value in history_track_pos[key]:
                                if abs(value) > 1:
                                    ticks_out_of_track += 1
                        norm_out_of_track_ticks = ticks_out_of_track/MAX_OUT_OF_TRACK_TICKS                    
                        
                        # compute the fitness for the current track
                        speed_comp_multiplier = 2
                        car_pos_multiplier = 2
                        fitness = (-normalized_avg_speed * speed_comp_multiplier) -normalized_distance_raced +normalized_damage +norm_out_of_track_ticks +\
                                    normalized_ticks + (norm_car_position * car_pos_multiplier)
                        # store the fitness for the current track
                        fitness_dict_component[track] = {
                                                          "fitness": fitness, "car_position": norm_car_position, 
                                                          "norm_avg_speed":-normalized_avg_speed,  "norm_distance_raced": -normalized_distance_raced,
                                                          "norm_damage": normalized_damage, "norm_out_of_track_ticks": norm_out_of_track_ticks,
                                                          "normalized_ticks": normalized_ticks, "sim_seconds": ticks/50
                                                        }
                        
                    else:
                        if race_failed:
                            print(f"RACE FAILED")
                        else:
                            print(f"THE AGENTS COULDN'T COMPLETE THE FIRST LAP")
                        fitness = 10   
                    #return fitness
                    
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    #print(message)
                    fitness = 20

                fitnesses_dict[track] = fitness
                self.fitness_terms[agent_indx] = fitness_dict_component
                
                """
                # check for constraint
                constraint = []
                for i in range(x.shape[0]):
                    constraint.append( int((x[i] < self.lb[i] or x[i] > self.ub[i])) )
                """

            # compute the average performance over all the tested tracks
            total_fitness = 0
            num_track = 0
            for fitness_on_track in fitnesses_dict.keys():
                total_fitness += fitnesses_dict[fitness_on_track]
                num_track += 1
            total_fitness /= num_track

            agents_cnt_lock.acquire(blocking=True)
            self.agents_cnt += 1
            print(f"Agent runned {self.agents_cnt}",end="\r")
            agents_cnt_lock.release()
            
            servers_port_state_lock.acquire(blocking=True)
            servers_port_state[port_number] = True
            servers_port_state_lock.notify_all()
            servers_port_state_lock.release()

            return total_fitness#, constraint
            
        # prepare the parameters for the pool
        params = []
        for k in range(len(X)):
            params.append((X[k],
                           k,
                           deepcopy(self.variable_to_change), 
                           deepcopy(self.controller_variables)))
           
        results = POOL.starmap(run_simulations, params)
        
        if adversarial == True:
            # take the MIN_TO_EVALUATE best agents
            indices = np.argpartition(results, MIN_TO_EVALUATE)[:MIN_TO_EVALUATE]
            # prepare the parameters for the pool
            params = []
            for indx in indices:
                params.append((X[indx],
                            indx,
                            deepcopy(self.variable_to_change), 
                            deepcopy(self.controller_variables)))
            
            best_agents_fitness_estimation = []
            best_agent_fitness_terms = [[] for i in range(NUM_RUN_FOR_BEST_EVALUATION)]
            # run NUM_RUN_FOR_BEST_EVALUATION simulation in order to better estimate
            # the fitness of the agent.
            for run in range(NUM_RUN_FOR_BEST_EVALUATION):
                best_agents_fitness_estimation.append(POOL.starmap(run_simulations, params))
                for agent in indices:
                    best_agent_fitness_terms[run].append(self.fitness_terms[agent])

            best_agents_fitness_estimation = np.array(best_agents_fitness_estimation)
            # compute the average fitness, for each agent
            for i, agent in enumerate(indices):
                # compute the average fitness for the given agent
                results[agent] = np.average(best_agents_fitness_estimation[:, i])
                
                # compute the average of the fitness term
                agent_fitness_term_avg = {}
                # for each track initialize the average terms.
                for track in track_names:
                    agent_fitness_term_avg[track] ={
                                                    "fitness": 0.0, "car_position": 0.0, 
                                                    "norm_avg_speed":-0.0,  "norm_distance_raced": -0.0,
                                                    "norm_damage": 0.0, "norm_out_of_track_ticks": 0.0,
                                                    "normalized_ticks": 0.0, "sim_seconds": 0.0
                                                    }

                # for each run of the agent
                for run in range(NUM_RUN_FOR_BEST_EVALUATION):
                    # fitness terms of the i-th agent, for the given run 
                    agent_fitness_term = best_agent_fitness_terms[run][i]
                    for track in agent_fitness_term:
                        # for each terms, for the given track.
                        for term in agent_fitness_term[track].keys():
                            agent_fitness_term_avg[track][term] += agent_fitness_term[track][term]
                
                for track in agent_fitness_term_avg:
                    for term in agent_fitness_term_avg[track].keys():
                        agent_fitness_term_avg[track][term] /= NUM_RUN_FOR_BEST_EVALUATION
                print(f"Agent {agent}, fitness terms {agent_fitness_term_avg}")
                self.fitness_terms[agent] = agent_fitness_term_avg

        """
        fitness = []
        constraints = []
        for i in range(len(results)):
            fitness.append(results[i][0])
            constraints.append(results[i][1])
        """ 
        out["F"] = np.array(results)
        #out["G"] = np.array(constraints)
        
        #print(f"Current solution fitness:\n{out['F']}")
        #print(f"Current solution constraing:\n{out['G']}")
        # best_fit = np.min(out["F"])
        best_fit_indx = np.argmin(out["F"])
        print(f"BEST FITNESS: {out['F'][best_fit_indx]}")
        best_fitness_terms = self.fitness_terms[best_fit_indx]
        for track in best_fitness_terms:
            print(f"Track {track}: {best_fitness_terms[track]}")
        

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


def create_population(n_pop, name_parameters_to_change):
    global population
    # initialize the population
    for i in range(n_pop):
        # for each parameter to change
        for j,key in enumerate(name_parameters_to_change):
            # compute the variation based on the default parameters
            variation = (PERCENTAGE_OF_VARIATION * parameters[key])/100
            operation = np.random.choice([0,1,2])
            offset = np.random.uniform(0, variation)
            if operation == 0:
                population[i][j] = parameters[key] + offset
            elif operation == 1:
                population[i][j] = parameters[key] - offset
            else:
                population[i][j] = parameters[key]
                
#'''
if __name__ == "__main__":
    ####################### SETUP ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', '-cp', help="checkpoint file containing the starting population for the algorithm", type= str,
                        default= "None")
    parser.add_argument('--configuration_file', '-conf', help="name of the configuration file to use, without extension or port number", type= str,
                    default= "quickrace_forza_no_adv")
    parser.add_argument('--controller_params', '-ctrlpar', help="initial controller parameters", type= str,
                    default= "Baseline_snakeoil\default_parameters")
    parser.add_argument('--param_change_cond_version', '-param_vers', help="path to the file containing the parameters to be changed by the algorithm", type= str,
                    default="parameter_change_condition_no_adv_v_2")
                    
    args = parser.parse_args()


    # load default parameters
    args.controller_params = '\\' + args.controller_params
    pfile= open(dir_path + args.controller_params,'r') 
    parameters = json.load(pfile)

    # load the change condition file
    parameters_to_change, change_cond_version = get_configuration(args.param_change_cond_version)
    
    track_names = take_track_names(args)

    tracks_folder = args.configuration_file
    results_folder = dir_path+"/Results_DE/"+ tracks_folder
    create_dir(results_folder)

    ####################### Differential Evolution ################################
    np_seed = 64
    de_seed = 100
    # set the np seed
    np.random.seed(np_seed)

    # Pymoo Differential Evolution
    print('Pymoo Differential Evolution')

    # compute the number of parameters to change
    # the lower and upper bound
    # name of parameter to change
    n_parameters = 0
    name_parameters_to_change = []
    lb = []
    ub = []
    for key in parameters_to_change:
        if parameters_to_change[key][0]:
            n_parameters += 1
            lb.append(parameters_to_change[key][1])
            ub.append(parameters_to_change[key][2])
            name_parameters_to_change.append(key)
    lb = np.array(lb)
    ub = np.array(ub)
    
    print(f"Number of parameters {n_parameters}")
    # population size
    n_pop = 5
    # number of variables for the problem visualization
    n_vars = n_parameters
    # maximum number of generations
    max_gens = 20
    # Cross-over rate
    cr = 0.9
    # Scaling factor F
    f = 1.5

    PARAMETERS_STRING = f"{np_seed}_{de_seed}_{n_pop}_{max_gens}_{n_vars}_{cr}_{f}_{PERCENTAGE_OF_VARIATION}_{change_cond_version}"

    # define the problem
    problem = TorcsProblem(variables_to_change = parameters_to_change, controller_variables = parameters, lb = lb, ub = ub)

    # Initializing the population randomly from the base controller or from the provided checkpoint file
    population = np.zeros((n_pop, n_vars))
    last_iteration = 0
    algorithm = None
    if args.checkpoint_file != "None":
        algorithm, last_iteration = load_checkpoint(args.checkpoint_file)
        if algorithm == None:
            sys.exit()
    else:
        create_population(n_pop, name_parameters_to_change)
        algorithm = DE(pop_size=n_pop, 
                    sampling= population,
                    variant="DE/rand/1/bin", 
                    CR=cr,
                    F=f,
                    dither="no",
                    jitter=False,
                    eliminate_duplicates=True)

        algorithm.setup(problem, ('n_gen', max_gens), seed=de_seed, verbose=True, save_history=True)

    for iter in range(last_iteration, max_gens):
        algorithm.next()

        res = algorithm.result()
        
        # save best result
        save_results(res.X[0])
        
        if SAVE_CHECKPOINT:
            checkpoint_file_name = save_checkpoint(algorithm, iter+1)
            algorithm , _ = load_checkpoint(checkpoint_file_name)
        algorithm.has_terminated = False
    
    res = algorithm.result()
    if res != None:
        print(f"final population fitness: {res.pop.get('F')}")
        print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

        # plot convergence
        n_evals = np.array([e.evaluator.n_eval for e in res.history])
        opt = np.array([e.opt[0].F for e in res.history])
        
        # save best result
        save_results(res.X[0])
        
        plt.title("Convergence")
        plt.plot(n_evals, opt, "-")
        #plt.yscale("log")
        file_name = results_folder  + "/" + PARAMETERS_STRING + '.png'
        plt.savefig(file_name)
        plt.show()
        
#'''