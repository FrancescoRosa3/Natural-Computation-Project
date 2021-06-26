from math import inf
import numpy as np
import matplotlib.pyplot as plt
import json, random, sys, threading, signal
from copy import deepcopy
import time
from multiprocessing.pool import ThreadPool

from numpy.lib.function_base import _select_dispatcher

from pymoo.algorithms.so_de import DE
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_visualization, get_termination
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

from pymoo.model.problem import Problem

# Load the custom_controller module
import custom_controller
# define the path were the parameters are defined
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# load default parameters
pfile= open(dir_path + "\Baseline_snakeoil\default_parameters",'r') 
parameters = json.load(pfile)

# load the change condition file
pfile= open(dir_path + "\parameter_change_condition",'r') 
parameters_to_change = json.load(pfile)

# CONSTANT DEFINITION
NUMBER_SERVERS = 10
BASE_PORT = 3000
PERCENTAGE_OF_VARIATION = 10

# CONSTANT FOR NORMALIZATION
EXPECTED_NUM_LAPS = 2
MAX_SPEED = 330
FORZA_LENGTH = 5784.10
FORZA_WIDTH = 11.0
WHEEL_LENGHT = 4328.54
WHEEL_WIDTH = 14.0
CG_1_LENGHT = 2057.56
CG_1_WIDTH = 15.0
UPPER_BOUND_DAMAGE = 1500
POOL = ThreadPool(NUMBER_SERVERS)
# ELEMENTS OF COST FUNCTION
cost_function = {}

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

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
        
        super().__init__(n_var=lb.shape[0], n_obj=1, n_constr=0, 
                         xl = lb, xu = ub)
                         #xl=np.array([-100000 for i in range(lb.shape[0])]), xu=np.array([100000 for i in range(lb.shape[0])]),
        self.variable_to_change = variables_to_change
        self.controller_variables = controller_variables
           
    # evaluate function
    def _evaluate(self, X, out, *args, **kwargs):
        
        def run_simulations(x, port_number, variable_to_change, controller_variables):
            i = 0
            for key in variable_to_change.keys():
                # change the value of contreller_variables
                # if the given variable is under evolution
                if variable_to_change[key][0] == 1:
                    # this parameter is under evolution
                    #print(f"key: {key} - starting value: {controller_variables[key]:.2f} - modified value: {x[agent_indx][i]}")
                    global lb, ub
                    
                    if x[i] <= lb[i] or x[i] >= ub[i]:
                        print(f"Overflow variabile {key}- Currente value {x[i]} -lb {lb[i]} - ub {ub[i]}")
                    
                    #x[i] = clip(x[i], lb[i], ub[i])
                    controller_variables[key] = x[i]
                    i += 1

            try:
                #print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                controller = custom_controller.CustomController(port=BASE_PORT+port_number+1,
                                                                parameters=controller_variables, 
                                                                parameters_from_file=False)
                
                history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos = controller.run_controller()
                # compute the number of laps
                num_laps = len(history_lap_time) 
                if num_laps > 0:
                    # compute the average speed
                    avg_speed = 0
                    for key in history_speed.keys():
                        avg_speed += np.average(history_speed[key])
                    avg_speed /= num_laps
                    normalized_avg_speed = avg_speed/MAX_SPEED
                
                    distance_raced = history_distance_raced[num_laps][-1]
                    normalized_distance_raced = distance_raced/(FORZA_LENGTH*EXPECTED_NUM_LAPS)
                
                    # take the damage
                    damage = history_damage[num_laps][-1]
                    normalized_damage = damage/UPPER_BOUND_DAMAGE
                
                    # compute the average from the center line
                    average_track_pos = 0
                    steps = 0
                    for key in history_track_pos.keys():
                        for value in history_track_pos[key]:
                            steps += 1
                            if abs(value) > 1:
                                average_track_pos += (abs(value) - 1)
                    average_track_pos /= steps
                    
                    #if damage > UPPER_BOUND_DAMAGE:
                    #    fitness[agent_indx] = np.inf
                    #else:
                    return (- normalized_avg_speed - normalized_distance_raced + normalized_damage + average_track_pos)
                    #print(F"SERVER: {indx} - distance: {distance_raced}")
                    #print(f"SERVER: {indx} - Fitness Value {fitness[agent_indx]:.2f}")
                    #print(f"Fitness Value {fitness[agent_indx]:.2f}\nNormalized AVG SPEED {normalized_avg_speed:.2f}\nNormalized Distance Raced {normalized_distance_raced:.2f}\nNormalized Damage {normalized_damage:.2f}\nAverage Track Pos {average_track_pos:.2f}")
                else:
                    '''           
                    # compute the average speed
                    avg_speed = 0
                    for key in history_speed.keys():
                        avg_speed += np.average(history_speed[key])
                    avg_speed /= len(history_speed[key])
                    normalized_avg_speed = avg_speed/MAX_SPEED

                    # compute the total distance raced
                    distance_raced = history_distance_raced[1][-1]
                    normalized_distance_raced = distance_raced/(FORZA_LENGTH)
                    # take the damage
                    damage = history_damage[1][-1]
                    normalized_damage = damage/UPPER_BOUND_DAMAGE
                    # compute the average from the center line
                    average_track_pos = 0
                    steps = 0
                    for key in history_track_pos.keys():
                        for value in history_track_pos[key]:
                            steps += 1
                            if abs(value) > 1:
                                average_track_pos += (abs(value) - 1)
                    average_track_pos /= steps
                    
                    fitness[agent_indx] = - normalized_avg_speed - normalized_distance_raced + normalized_damage + average_track_pos
                    #print(F"SERVER: {indx} - distance: {distance_raced}")
                    print(f"THE AGENTS COULDN'T COMPLETE THE FIRST LAP")    
                    #print(f"SERVER: {indx} - Fitness Value {fitness[agent_indx]:.2f}")
                    # fitness[agent_indx] = np.inf
                    '''
                    return np.inf
            except:
                    print(f"Exception")
                    return np.inf

        # prepare the parameters for the pool
        port_number = 0
        params = []
        for k in range(len(X)):
            print(port_number)
            params.append((X[k], 
                           port_number%NUMBER_SERVERS,
                           deepcopy(self.variable_to_change), 
                           deepcopy(self.controller_variables)))
            port_number += 1
        
        F = POOL.starmap(run_simulations, params)
        out["F"] = np.array(F)

if __name__ == "__main__":

    np_seed = 0
    de_seed = 123
    # set the np seed
    np.random.seed(np_seed)

    # Pymoo Differential Evolution
    print_hi('Pymoo Differential Evolution')

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
    n_pop = 
    # number of variables for the problem visualization
    n_vars = n_parameters
    # maximum number of generations
    max_gens = 10
    # Cross-over rate
    cr = 0.6
    # Scaling factor F
    f = 0.9

    # initialize the population
    population = np.zeros((n_pop, n_vars))
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
            #print(f"PARAMETER: {key}: {parameters[key]} - variation: {variation} - final_value: {population[i][j]}")


    # define the problem
    problem = TorcsProblem(variables_to_change = parameters_to_change, controller_variables = parameters, lb = lb, ub = ub)
    # define the termination criteria
    termination = get_termination("n_gen", max_gens)
        
    algorithm = DE(pop_size=n_pop, 
                   sampling= population,
                   variant="DE/rand/1/bin", 
                   CR=cr,
                   F=f,
                   dither="no", 
                   jitter=False,
                   eliminate_duplicates=True)

    res = minimize(problem, algorithm, termination, seed=de_seed, verbose=True, save_history=True)
    print(f"final population fitness: {res.pop.get('F')}")
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    # plot convergence
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    
    # save best result
    print("Saving the best result on file.....")
    i = 0
    for key in parameters_to_change.keys():
        # change the value of contreller_variables
        # if the given variable is under evolution
        if parameters_to_change[key][0] == 1:
            # this parameter is under evolution
            parameters[key] = res.X[i]
            i += 1
    file_name = dir_path+"/Results/"+"Forza/"+f"{np_seed}_{de_seed}_{n_pop}_{n_vars}_{cr}_{f}.xml"
    with open(file_name, 'w') as outfile:
        json.dump(parameters, outfile)
    
    plt.title("Convergence")
    plt.plot(n_evals, opt, "-")
    #plt.yscale("log")
    plt.show()