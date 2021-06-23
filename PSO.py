# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
import sys
# Change directory to access the pyswarms module
sys.path.append('../')

rc('animation', html='html5')

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# function to optimize
def func():
    pass

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
problem_size = 2
swarm_size = 50
iterations = 1000

# Call instance of PSO
optimizer = ps.single.LocalBestPSO(n_particles=swarm_size, dimensions=problem_size, options=options)

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