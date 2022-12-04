
# Project Title
SRIP IITGN 2022 - Machine Learning - Prof. Nipun Batra

## Tasks
1. **Animate bivariate normal distribution.** [10 Marks]

    https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/MultivariateNormal.png/793px-MultivariateNormal.png?20130322193052

Reproduce the above figure showing samples from bivariate normal with marginal PDFs from scratch using JAX and matplotlib.

Add interactivity to the figure by adding sliders with ipywidgets. You should be able to vary the parameters of bivariate normal distribution (mean and covariance matrix) using ipywidgets.

2. **Implement from scratch a sampling method to draw samples from a multivariate Normal (MVN) distribution in JAX.** [10 Marks]

Your code should work for any number of dimensions but please set the number of dimensions (random variables of MVN) to 10 for this task.

You are only allowed to use jax.random.uniform.You are especially not allowed to use jax.random.normal.

You should randomly create the mean and covariance matrix to fully specify an MVN distribution.

Implement a sampling method from scratch using which you can draw samples from the specified MVN distribution.

Use your sampling method to draw multiple samples from the MVN distribution and reconstruct 
the parameters of your MVN distribution (mean and covariance matrix) to confirm that 
your sampling method is working correctly.

## How to run the code
There are two ipynb files in the repository.The
file named **Animated_bivariate.ipynb** contains the code
for the task-1 and the other file named **sampling_from_MVN.ipynb**
conatins the code for the task-2.

To run the files for a given task , download 
the corresponding .ipynb file, upload it in a 
suitable editor and run the cells.

In the file **sampling_from_MVN.ipynb** to change 
the dimensions , change the value of the variable **`dim`** 
from 10 to a new value.



