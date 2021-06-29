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
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET
from threading import Lock, Condition
from copy import deepcopy
import argparse

# Change directory to access the pyswarms module
sys.path.append('../')

#rc('animation', html='html5')

# Load the custom_controller module
import custom_controller
# define the path were the parameters are defined
import os 

# CONSTANT DEFINITION
NUMBER_SERVERS = 10
BASE_PORT = 3000
PERCENTAGE_OF_VARIATION = 20

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
MAX_OUT_OF_TRACK_TICKS = 1000       # corresponds to 20 sec
NUMBER_AVAILABLE_POSITION = 9

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

#from pyswarms.utils.functions import single_obj as fx

# class that defines the problem to solve
class TorcsProblem():

    # Problem initialization
    def __init__(self, variables_to_change, controller_variables, lb, ub):
        
        self.variable_to_change = variables_to_change
        self.controller_variables = controller_variables
        self.lb = lb
        self.ub = ub

        global swarm_size
        self.agents_cnt = 0
        self.fitness_terms = [None for i in range(swarm_size)]
           
    # evaluate function
    def evaluate(self, X):
        
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
                    
                    history_lap_time, history_speed, history_damage, history_distance_raced, history_track_pos, history_car_pos, ticks = controller.run_controller()
                    
                    normalized_ticks = ticks/controller.C.maxSteps

                    # compute the number of laps
                    num_laps = len(history_lap_time)

                    # the car has completed at least the first lap
                    if num_laps > 0:
                        # compute the average speed
                        avg_speed = 0
                        for history_key in history_speed.keys():
                            for value in history_speed[history_key]:
                                avg_speed += value
                        avg_speed /= ticks
                        #print(f"Num Laps {num_laps} - Average Speed {avg_speed} - Num ticks {ticks}")
                        
                        normalized_avg_speed = avg_speed/MAX_SPEED

                        distance_raced = history_distance_raced[history_key][-1]
                        normalized_distance_raced = distance_raced/(TRACK_LENGTH[track]*EXPECTED_NUM_LAPS)
                    
                        # take the damage
                        damage = history_damage[history_key][-1]
                        normalized_damage = damage/UPPER_BOUND_DAMAGE

                        # take the car position at the end of the race
                        car_position = history_car_pos[history_key][-1]
                        norm_car_position = car_position/NUMBER_AVAILABLE_POSITION

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
                        fitness = -normalized_avg_speed * speed_comp_multiplier -normalized_distance_raced +normalized_damage +norm_out_of_track_ticks +normalized_ticks +norm_car_position
                        # store the fitness for the current track
                        fitness_dict_component[track] = f"Fitness {fitness}-Car position {norm_car_position}- Norm AVG SPEED {-normalized_avg_speed}- Norm Distance Raced {-normalized_distance_raced}-Norm Damage {normalized_damage}- norm out_of_track_ticks {norm_out_of_track_ticks}- normalized ticks {normalized_ticks}- Sim seconds {ticks/50}"
                        
                    else:
                        #print(f"THE AGENTS COULDN'T COMPLETE THE FIRST LAP")
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
            print(f"Agent runned {self.agents_cnt}", end="\r")
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
        
        """
        fitness = []
        constraints = []
        for i in range(len(results)):
            fitness.append(results[i][0])
            constraints.append(results[i][1])
        """ 
        fitness = np.array(results)
        #out["G"] = np.array(constraints)
        
        #print(f"Current solution fitness:\n{fitness}")
        #print(f"Current solution constraing:\n{out['G']}")
        # best_fit = np.min(out["F"])
        best_fit_indx = np.argmin(fitness)
        print(f"BEST FITNESS: {fitness[best_fit_indx]}")
        best_fitness_terms = self.fitness_terms[best_fit_indx]
        for track in best_fitness_terms:
            print(f"Track {track}: {best_fitness_terms[track]}")

        return fitness


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

def create_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def create_population(n_pop, name_parameters_to_change):
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
            population[i][j] = clip(population[i][j], lb[j], ub[j])
            #print(f"PARAMETER: {key}: {parameters[key]} - variation: {variation} - final_value: {population[i][j]}")
    return population


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load default parameters
    pfile= open(dir_path + "\Baseline_snakeoil\default_parameters",'r') 
    parameters = json.load(pfile)

    # load the change condition file
    pfile= open(dir_path + "\parameter_change_condition",'r') 
    parameters_to_change = json.load(pfile)

    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration_file', '-conf', help="name of the configuration file to use, without extension or port number", type= str,
                    default= "quickrace_forza_no_adv")
                    
    args = parser.parse_args()
    track_names = take_track_names(args)

    np_seed = 0
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

    # Set-up hyperparameters
    #options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    options = {'c1': 1, 'c2': 0.8, 'w': 0.7298, 'k': 10, 'p': 2}
    problem_size = n_parameters
    swarm_size = 50
    iterations = 10

    PARAMETERS_STRING = f"{np_seed}_{swarm_size}_{iterations}_{n_parameters}_{options['c1']}_{options['c2']}_{options['w']}_{options['k']}_{options['p']}_{PERCENTAGE_OF_VARIATION}"

    tp = TorcsProblem(variables_to_change=parameters_to_change, controller_variables=parameters, lb=lb, ub=ub)

    population = np.zeros((swarm_size, n_parameters))
    create_population(swarm_size, name_parameters_to_change)

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
    
    tracks_folder = args.configuration_file
    results_folder = dir_path + "/Results_PSO/" + tracks_folder
    create_dir(results_folder)
    
    file_name = results_folder  + "/" + PARAMETERS_STRING + ".xml"
    
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