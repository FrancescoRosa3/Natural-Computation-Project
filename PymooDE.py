import numpy as np
import matplotlib.pyplot as plt
import json, random, sys

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

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# class that defines the problem to solve
class TORCS_PROBLEM(Problem):
    
    # Problem initialization
    def __init__(self, parameters):
            xl = np.zeros(n_parameters)
            xu = np.zeros(n_parameters)
            for i in range(xl.size):
                xl[i] = -100000
                xu[i] = 100000
            super().__init__(n_var=n_parameters, n_obj=1, n_constr=0, xl=xl, xu=xu, elementwise_evaluation = True)
            self.parameters = parameters
    
    # evaluate function
    # we take in input a single individual
    # run the experiment
    # take the result
    def _evaluate(self, x, out, *args, **kwargs):
        # counter for the attribute vector x
        i = 0
        for k in self.parameters.keys():
            # assign the current parameter value
            self.parameters[k] = x[i]
            print(x[i])
            i += 1
        try:
            
            history_lap_time, history_speed, history_damage = custom_controller.run_controller(parameters=self.parameters, 
                                                                                                parameters_from_file=False)
            average_time = 0.0
            cnt = 0
            for key in history_lap_time.keys():
                if key > 1:
                    average_time += history_lap_time[key]
                    cnt += 1
            if cnt != 0:
                out["F"] = average_time/cnt
            else:
                out["F"] = np.inf
        except KeyboardInterrupt:
            sys.exit()
        except:
            out["F"] = np.inf


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
    n_pop = 5
    # number of variables for the problem visualization
    n_vars = n_parameters
    # maximum number of generations
    max_gens = 100
    # Cross-over rate
    cr = 0.9
    # Scaling factor F
    f = 0.9

    # define the problem
    problem = TORCS_PROBLEM(parameters = parameters)
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
                   eliminate_duplicates=False)

    res = minimize(problem, algorithm, termination, seed=112, verbose=True, save_history=True)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    # plot convergence
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])

    plt.title("Convergence")
    plt.plot(n_evals, opt, "-")
    plt.yscale("log")
    plt.show()