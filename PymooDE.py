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

import argparse

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
MAX_SPEED = 300
FORZA_LENGTH = 5784.10
FORZA_WIDTH = 11.0
WHEEL_LENGHT = 4328.54
WHEEL_WIDTH = 14.0
CG_1_LENGHT = 2057.56
CG_1_WIDTH = 15.0
UPPER_BOUND_DAMAGE = 1500
MAX_OUT_OF_TRACK_TICKS = 1000       # corresponds to 20 sec
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
                        xl=np.array([-100000 for i in range(lb.shape[0])]), xu=np.array([100000 for i in range(lb.shape[0])]))#,xl = lb, xu = ub)
                         
        self.variable_to_change = variables_to_change
        self.controller_variables = controller_variables

        self.fitness_terms = {}
           
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
                    '''
                    if x[i] <= lb[i] or x[i] >= ub[i]:
                        print(f"Overflow variabile {key}- Currente value {x[i]} -lb {lb[i]} - ub {ub[i]}")
                    '''
                    #x[i] = clip(x[i], lb[i], ub[i])
                    controller_variables[key] = x[i]
                    i += 1

            try:
                #print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                controller = custom_controller.CustomController(port=BASE_PORT+port_number+1,
                                                                parameters=controller_variables, 
                                                                parameters_from_file=False)
                
                history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, ticks = controller.run_controller()
                
                normalized_ticks = ticks/controller.C.maxSteps

                # compute the number of laps
                num_laps = len(history_lap_time)

                if num_laps > 0:
                    # compute the average speed
                    avg_speed = 0
                    for key in history_speed.keys():
                        for value in history_speed[key]:
                            avg_speed += value
                    avg_speed /= ticks
                    #print(f"Num Laps {num_laps} - Average Speed {avg_speed} - Num ticks {ticks}")
                    
                    normalized_avg_speed = avg_speed/MAX_SPEED
                
                    distance_raced = history_distance_raced[num_laps][-1]
                    normalized_distance_raced = distance_raced/(FORZA_LENGTH*EXPECTED_NUM_LAPS)
                
                    # take the damage
                    damage = history_damage[num_laps][-1]
                    normalized_damage = damage/UPPER_BOUND_DAMAGE
                
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
                                
                    fitness = -normalized_avg_speed -normalized_distance_raced +normalized_damage +norm_out_of_track_ticks +normalized_ticks
                    self.fitness_terms[fitness] = {"Norm AVG SPEED": -normalized_avg_speed, "Norm Distance Raced": -normalized_distance_raced, "Norm Damage": normalized_damage, "norm out_of_track_ticks": norm_out_of_track_ticks, "normalized ticks": normalized_ticks, "Sim seconds": ticks/50}
                    
                else:
                    print(f"THE AGENTS COULDN'T COMPLETE THE FIRST LAP")
                    fitness = np.inf  
                return fitness
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                return np.inf
                
                
        # prepare the parameters for the pool
        port_number = 0
        params = []
        for k in range(len(X)):
            params.append((X[k], 
                           port_number%NUMBER_SERVERS,
                           deepcopy(self.variable_to_change), 
                           deepcopy(self.controller_variables)))
            port_number += 1
        
        F = POOL.starmap(run_simulations, params)
        out["F"] = np.array(F)
        
        print(out["F"])
        best_fit = np.min(out["F"])
        if best_fit != np.inf:
            print(f"BEST FITNESS: {best_fit} - terms: {self.fitness_terms[best_fit]}")

def create_checkpoint_dir(checkpoint_folder):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

def save_checkpoint(algorithm, iter):
    global np_seed, de_seed, n_pop, max_gens, n_vars, cr, f
    checkpoint_folder = dir_path + "/Checkpoints"+ f"/{np_seed}_{de_seed}_{n_pop}_{max_gens}_{n_vars}_{cr}_{f}"
    create_checkpoint_dir(checkpoint_folder)
    checkpoint_file_name = checkpoint_folder + f"/{np_seed}_{de_seed}_{n_pop}_{max_gens}_{n_vars}_{cr}_{f}_pop-iter-{iter}.npy"
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

def create_population(n_pop, n_vars, name_parameters_to_change):
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
            #print(f"PARAMETER: {key}: {parameters[key]} - variation: {variation} - final_value: {population[i][j]}")
    return population


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', '-cf', help="checkpoint file containing the starting population for the algorithm", type= str,
                        default= "None")
    args = parser.parse_args()

    np_seed = 1
    de_seed = 124
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
    n_pop = 100
    # number of variables for the problem visualization
    n_vars = n_parameters
    # maximum number of generations
    max_gens = 20
    # Cross-over rate
    cr = 0.7
    # Scaling factor F
    f = 0.9

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
        create_population(n_pop, n_vars, name_parameters_to_change)
        algorithm = DE(pop_size=n_pop, 
                    sampling= population,
                    variant="DE/rand/1/bin", 
                    CR=cr,
                    F=f,
                    dither="no",
                    jitter=False,
                    eliminate_duplicates=True)

    res = None
    for iter in range(last_iteration, max_gens):
        res = minimize(problem, algorithm, ('n_gen', 1), seed=de_seed, verbose=True, save_history=True)
        checkpoint_file_name = save_checkpoint(algorithm, iter+1)
        algorithm, last_iteration = load_checkpoint(checkpoint_file_name)
        algorithm.has_terminated = False

    if res != None:
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
        file_name = dir_path+"/Results/"+"Forza/"+f"{np_seed}_{de_seed}_{n_pop}_{max_gens}_{n_vars}_{cr}_{f}.xml"
        with open(file_name, 'w') as outfile:
            json.dump(parameters, outfile)
        
        plt.title("Convergence")
        plt.plot(n_evals, opt, "-")
        #plt.yscale("log")
        plt.show()

'''
if __name__ == "__main__":
    par = {'dnsh3rpm': 7043.524501178322, 'consideredstr8': 0.010033102875417081, 'upsh6rpm': 14789.41256679235, 'dnsh2rpm': 7062.974503018889, 'str8thresh': 0.14383741216415255, 'safeatanyspeed': 0.0012800919243947557, 'offroad': 1.0002653228126588, 'fullstmaxsx': 20.070818862674596, 'wwlim': 4.487048259980536, 'upsh4rpm': 9850.06473842414, 'dnsh1rpm': 4086.4281437604054, 'sxappropriatest1': 16.08326982212498, 'oksyp': 0.06586706491973668, 'slipdec': 0.018047798233552067, 'spincutslp': 0.05142589453207698, 'ignoreinfleA': 10.793558810628733, 's2sen': 3.4996561397948263, 'dnsh4rpm': 7474.1667888177535, 'seriousABS': 30.237506792415605, 'dnsh5rpm': 8106.559743640354, 'obviousbase': 93.7374985607152, 'stst': 513.5298776779491, 'upsh3rpm': 9520.747330115491, 'clutch_release': 0.050017716984125785, 'stC': 297.23435114322194, 'pointingahead': 2.196445074922305, 'spincutclip': 0.10803464385693813, 'clutchslip': 90.34100694329528, 'obvious': 1.3232214861047789, 'backontracksx': 69.09088237263713, 'upsh2rpm': 10116.745791908856, 'senslim': 0.031006049843539912, 'clutchspin': 50.291035172311716, 'fullstis': 0.7759314134954662, 'brake': 0.5230215749291802, 'carmin': 34.98602774087677, 'sycon2': 1.000239198433143, 's2cen': 0.4701703131364017, 'sycon1': 0.6429244177717478, 'upsh5rpm': 9309.101130251716, 'carmaxvisib': 2.317604406633401, 'sxappropriatest2': 0.5520202884372154, 'skidsev1': 0.576616694479842, 'wheeldia': 0.8554277668777635, 'brakingpacefast': 1.0083267830865785, 'sensang': -0.7558273506842333, 'spincutint': 1.7810420061006913, 'st': 648.0720326063517, 'brakingpaceslow': 2.0841388650800172, 'sortofontrack': 1.5040683093640903, 'steer2edge': 0.9193250804761118, 'backward': 1.5085149570615013}
    file_name = dir_path+"/Results/"+"Forza/"+f"sgaso.xml"
    with open(file_name, 'w') as outfile:
        json.dump(parameters, outfile)
'''