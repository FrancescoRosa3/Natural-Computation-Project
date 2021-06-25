from math import inf
import numpy as np
import matplotlib.pyplot as plt
import json, random, sys, threading, signal
from copy import deepcopy
import xml.etree.ElementTree as ET
import argparse

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
NUMBER_SERVERS = 1
BASE_PORT = 3000
PERCENTAGE_OF_VARIATION = 20

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

# list of track names
track_names = []

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# class that defines the problem to solve
class TorcsProblem(Problem):

    # Problem initialization
    def __init__(self, variables_to_change, controller_variables, lb, ub):
            super().__init__(n_var=lb.shape[0], n_obj=1, n_constr=0, xl=lb, xu=ub)
            self.variable_to_change = variables_to_change
            self.controller_variables = controller_variables
    

    def run_simulations(indx, num_individuals_to_run, fitness, x, variable_to_change, controller_variables):
        global track_names

        if indx != NUMBER_SERVERS - 1:
            # compute the start agent index
            start_agent_indx = num_individuals_to_run * indx
            # compute the end agent index
            end_agent_indx = start_agent_indx + num_individuals_to_run
        else:
            # compute the start agent index
            start_agent_indx = x.shape[0]-num_individuals_to_run
            # compute the end agent index
            end_agent_indx = x.shape[0]

        # for each agent that the thread must run
        for agent_indx in range(start_agent_indx, end_agent_indx):

            i = 0
            for key in variable_to_change.keys():
                # change the value of contreller_variables
                # if the given variable is under evolution
                if variable_to_change[key][0] == 1:
                    # this parameter is under evolution
                    controller_variables[key] = x[agent_indx][i]
                    i += 1
            for track in track_names:   
                print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1} on Track {track}")
                try:
                    #print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                    controller = custom_controller.CustomController(port=BASE_PORT+indx+1,
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

                        # compute the total distance raced
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
                        fitness[agent_indx] = - normalized_avg_speed - normalized_distance_raced + normalized_damage + average_track_pos
                        #print(f"Fitness Value {fitness[agent_indx]}\nNormalized AVG SPEED {normalized_avg_speed}\nNormalized Distance Raced {normalized_distance_raced}\nNormalized Damage {normalized_damage}\nAverage Track Pos {average_track_pos}")
                    else:
                        fitness[agent_indx] = np.inf
                except:
                    #print(f"Exception")
                    fitness[agent_indx] = np.inf

    # evaluate function
    def _evaluate(self, x, out, *args, **kwargs):

        # list of fitness values
        fitness = [np.inf for i in range(x.shape[0])]
        # compute the number of agents that each thread must run
        number_of_individuals_per_thread, remainder = divmod(x.shape[0], NUMBER_SERVERS)
        # list of thread
        threads = []
        for i in range(NUMBER_SERVERS):
            # check for the last thread
            # assign the reaminder number of individuals to the last thread
            if i == NUMBER_SERVERS - 1:
                num_individuals_to_run = number_of_individuals_per_thread + remainder
            else:
                num_individuals_to_run = number_of_individuals_per_thread

            threads.append(threading.Thread(target=TorcsProblem.run_simulations, 
                                            args=(i, num_individuals_to_run, fitness, x, deepcopy(self.variable_to_change), deepcopy(self.controller_variables)), daemon = True))
            
            # run the i-th thread
            threads[i].start()

        # wait for all thread to end
        for i in range(NUMBER_SERVERS):
            threads[i].join()

        out["F"] = np.array(fitness).T
        print(out["F"])

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration_file', '-cf', help="name of the configuration file to use, without extension or port number", type= str,
                    default= "quickrace_forza_no_adv")
                    
    args = parser.parse_args()
    
    track_names = take_track_names(args)

    # set the np seed
    np.random.seed(0)

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
    n_pop = 3
    # number of variables for the problem visualization
    n_vars = n_parameters
    # maximum number of generations
    max_gens = 2
    # Cross-over rate
    cr = 0.9
    # Scaling factor F
    f = 0.2

    # initialize the population
    population = np.zeros((n_pop, n_vars))
    for i in range(n_pop):
        # for each parameter to change
        for j,key in enumerate(name_parameters_to_change):
            # compute the variation based on the default parameters
            variation = (PERCENTAGE_OF_VARIATION * parameters[key])/100
            operation = np.random.choice([0,1,2])
            if operation == 0:
                population[i][j] = parameters[key] + variation
            elif operation == 1:
                population[i][j] = parameters[key] - variation
            else:
                population[i][j] = parameters[key]

    # define the problem
    problem = TorcsProblem(variables_to_change = parameters_to_change, controller_variables = parameters, lb = lb, ub = ub)
    # define the termination criteria
    termination = get_termination("n_gen", max_gens)
        
    algorithm = DE(pop_size=n_pop, 
                   sampling= population,
                   variant="DE/rand/1/bin", 
                   CR=cr,
                   F=f,
                   dither="vector", 
                   jitter=True,
                   eliminate_duplicates=True)

    res = minimize(problem, algorithm, termination, seed=112, verbose=True, save_history=True)
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
    
    file_name = dir_path+"/Results/"+"Forza/"+f"{n_pop}_{n_vars}_{cr}_{f}.xml"
    with open(file_name, 'w') as outfile:
        json.dump(parameters, outfile)
    
    plt.title("Convergence")
    plt.plot(n_evals, opt, "-")
    #plt.yscale("log")
    plt.show()


 