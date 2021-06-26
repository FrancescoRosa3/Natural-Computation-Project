# Import modules
# Import PySwarms
import pyswarms as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
import sys
import json
import threading
from copy import deepcopy
from multiprocessing.pool import ThreadPool
# Change directory to access the pyswarms module
sys.path.append('../')

#rc('animation', html='html5')

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
PERCENTAGE_OF_VARIATION = 20
POOL = ThreadPool(NUMBER_SERVERS)

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

#from pyswarms.utils.functions import single_obj as fx

# class that defines the problem to solve
class TorcsProblem():

    # Problem initialization
    def __init__(self, variables_to_change, controller_variables, lb, ub):                 
        self.variable_to_change = variables_to_change
        self.controller_variables = controller_variables

        self.fitness_terms = {}
           
    # evaluate function
    def evaluate(self, X):
        
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

                    # compute #ticks in which the vehicle drive at a speed lower then avarage speed
                    temp_ticks = 0
                    avg_negative_speed_variation = 0
                    for key in history_speed.keys():
                        for value in history_speed[key]:
                            if value < avg_speed:
                                temp_ticks += 1
                                avg_negative_speed_variation += (avg_speed - value)
                    avg_negative_speed_variation /= temp_ticks
                    norm_avg_negative_speed_variation = avg_negative_speed_variation / ticks
                                
                    fitness = -normalized_avg_speed -normalized_distance_raced +normalized_damage +norm_out_of_track_ticks + normalized_ticks
                    self.fitness_terms[fitness] = {"Norm AVG SPEED": -normalized_avg_speed, "Norm Distance Raced": -normalized_distance_raced, "Norm Damage": normalized_damage,
                        "norm out_of_track_ticks": norm_out_of_track_ticks, "normalized ticks": normalized_ticks, "Sim seconds": ticks/50}
                    
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
        
        res = np.array(F)
        print(res)
        best_fit = np.min(res)
        if best_fit != np.inf:
            print(f"BEST FITNESS: {best_fit} - terms: {self.fitness_terms[best_fit]}")

        return res


# function to optimize
def func(x):
    global tp
    return tp.evaluate(x)

def clip(param, lb, ub):
    if param < lb:
        return lb
    if param > ub:
        return ub
    return param

if __name__ == '__main__':

    np_seed = 0

    # set the np seed
    np.random.seed(np_seed)

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

    tp = TorcsProblem(variables_to_change=parameters_to_change, controller_variables=parameters, lb=lb, ub=ub)

    # Set-up hyperparameters
    #options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    options = {'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298, 'k': 2, 'p': 2}
    problem_size = n_parameters
    swarm_size = 3
    iterations = 1 #1000

    # initialize the population
    population = np.zeros((swarm_size, n_parameters))
    for i in range(swarm_size):
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
            population[i][j] = clip(population[i][j], lb[j], ub[j])

    '''
    for i in range(swarm_size):
        # for each parameter to change
        for j,key in enumerate(name_parameters_to_change):
            print("TRUE" if parameters[key]>=lb[j] and parameters[key]<=ub[j] else "FALSE", end=' ')
            print(f"{key=} {parameters[key]=} {lb[j]=} {ub[j]=}")
    '''

    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=swarm_size, dimensions=problem_size, options=options, init_pos=population, bounds=(lb,ub))

    # Perform optimization
    cost, pos = optimizer.optimize(func, iters=iterations)

    print(cost)

    # save best result
    print("Saving the best result on file.....")
    i = 0
    for key in parameters_to_change.keys():
        # change the value of contreller_variables
        # if the given variable is under evolution
        if parameters_to_change[key][0] == 1:
            # this parameter is under evolution
            parameters[key] = pos[i]
            i += 1
    file_name = dir_path+"/Results/"+"Forza/"+f"PSO{np_seed}_{options['c1']}_{options['c2']}_{options['w']}_{swarm_size}_{iterations}.xml"
    with open(file_name, 'w') as outfile:
        json.dump(parameters, outfile)

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    '''
    if(problem_size == 2):
        animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))
        # Enables us to view it in a Jupyter notebook
        HTML(animation.to_html5_video())
        animation.save('dynamic_images.mp4')
    '''