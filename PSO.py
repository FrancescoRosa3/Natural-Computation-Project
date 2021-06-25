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

#from pyswarms.utils.functions import single_obj as fx

# class that defines the problem to solve
class TorcsProblem():

    # Problem initialization
    def __init__(self, variables_to_change, controller_variables, lb, ub):
        self.variable_to_change = variables_to_change
        self.controller_variables = controller_variables
    

    def run_simulations(indx, num_individuals_to_run, fitness, x, variable_to_change, controller_variables):
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

            print(f"AGENT PARAMS: {controller_variables}")
                
            try:
                print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
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
                    print(f"Fitness Value {fitness[agent_indx]}\nNormalized AVG SPEED {normalized_avg_speed}\nNormalized Distance Raced {normalized_distance_raced}\nNormalized Damage {normalized_damage}\nAverage Track Pos {average_track_pos}")
                else:
                    fitness[agent_indx] = np.inf
            except:
                #print(f"Exception")
                fitness[agent_indx] = np.inf

    # evaluate function
    def evaluate(self, x):

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

        return np.array(fitness).T

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

    # set the np seed
    np.random.seed(0)

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
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    problem_size = n_parameters
    swarm_size = 50
    iterations = 1000

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

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    '''
    if(problem_size == 2):
        animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, mark=(0, 0))
        # Enables us to view it in a Jupyter notebook
        HTML(animation.to_html5_video())
        animation.save('dynamic_images.mp4')
    '''