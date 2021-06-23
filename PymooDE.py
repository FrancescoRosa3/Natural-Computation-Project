from math import inf
import numpy as np
import matplotlib.pyplot as plt
import json, random, sys, threading, signal
from copy import deepcopy

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
n_parameters = len(parameters)

# CONSTANT DEFINITION
NUMBER_SERVERS = 10
BASE_PORT = 3000

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# class that defines the problem to solve
class TorcsProblem(Problem):

    # Problem initialization
    def __init__(self, parameters):
            xl = np.zeros(n_parameters)
            xu = np.zeros(n_parameters)
            for i in range(xl.size):
                xl[i] = -100000
                xu[i] = 100000
            super().__init__(n_var=n_parameters, n_obj=1, n_constr=0, xl=xl, xu=xu)
            self.parameters_dict = parameters
    

    def run_simulations(indx, num_agents_to_run, fitness, x, parameters_dict):
        if indx != NUMBER_SERVERS - 1:
            # compute the start agent index
            start_agent_indx = num_agents_to_run * indx
            # compute the end agent index
            end_agent_indx = start_agent_indx + num_agents_to_run
        else:
            # compute the start agent index
            start_agent_indx = x.shape[0]-num_agents_to_run
            # compute the end agent index
            end_agent_indx = x.shape[0]

        # for each agent that the thread must run
        for agent_indx in range(start_agent_indx, end_agent_indx):

            # assign to the parameter dict the parameter value for the agent_indx-th agent
            # counter for the attribute vector x
            i = 0
            for k in parameters_dict.keys():
                # assign the current parameter value
                parameters_dict[k] = x[agent_indx][i]
                #print(f"\"{k}\": {self.parameters[k]},")
                i += 1
            try:
                print(f"Run agent {agent_indx} on Port {BASE_PORT+indx+1}")
                controller = custom_controller.CustomController(port=BASE_PORT+indx+1,
                                                                parameters=parameters_dict, 
                                                                parameters_from_file=False)
                history_lap_time, history_speed, history_damage = controller.run_controller()

                average_time = 0.0
                cnt = 0
                for key in history_lap_time.keys():
                    if key > 1:
                        average_time += history_lap_time[key]
                        cnt += 1
                if cnt != 0:
                    fitness[agent_indx] = average_time/cnt
                else:
                    fitness[agent_indx] = np.inf
            except (RuntimeWarning, RuntimeError) as e:
                print(f"Exception {e}")
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
                                            args=(i, num_individuals_to_run, fitness, x, deepcopy(self.parameters_dict)), daemon = True))
            
            # run the i-th thread
            threads[i].start()

        # wait for all thread to end
        for i in range(NUMBER_SERVERS):
            threads[i].join()

        out["F"] = np.array(fitness).T
        print(out["F"])

# simple n_ary tournament for a single-objective algorithm
def n_ary_tournament(pop, P, algorithm, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors > pop.size:
        raise Exception(" Max pressure greater than pop.size not allowed for tournament!")

    # the result this function returns
    import numpy as np
    from random import random
    S = np.full(n_tournaments, -1, dtype=int)

    # now do all the tournaments
    for i in range(n_tournaments):
        selected = P[i]
        for j in range(n_competitors):
            # if the first individiual is better, choose it
            if pop[selected[j]].F < pop[selected[j]].F:
                winner = selected[j]
            # otherwise take the other individual
            else:
                winner = selected[j]
        S[i] = winner
    return S


if __name__ == "__main__":
    # Pymoo Differential Evolution
    print_hi('Pymoo Differential Evolution')

    # population size
    #n_pop = 5*n_parameters
    n_pop = 10
    # number of variables for the problem visualization
    n_vars = n_parameters
    # maximum number of generations
    max_gens = 3
    # Cross-over rate
    cr = 0.9
    # Scaling factor F
    f = 0.9

    # define the problem
    problem = TorcsProblem(parameters = parameters)
    # define the termination criteria
    termination = get_termination("n_gen", max_gens)

    # create the starting population
    parameters_start_values = []
    for value in parameters.values():
        parameters_start_values.append(value)
    parameters_start_values = np.array(parameters_start_values)

    population = np.zeros((n_pop, n_vars))
    for i in range(n_pop):
        for j in range(parameters_start_values.size):
            population[i][j] = parameters_start_values[j] + np.random.uniform(-1.0, 1.0)
        
    algorithm = DE(pop_size=n_pop, 
                   sampling= population,
                   variant="DE/rand/1/bin", 
                   CR=cr,
                   F=f,
                   dither="vector", 
                   jitter=True,
                   eliminate_duplicates=True)

    res = minimize(problem, algorithm, termination, seed=112, verbose=True, save_history=True)
    print(res.history)
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    # plot convergence
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])

    plt.title("Convergence")
    plt.plot(n_evals, opt, "-")
    plt.yscale("log")
    plt.show()