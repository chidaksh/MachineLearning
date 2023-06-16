# Overview
[line_search.py](line_search.py) is a program which compares the performance of gradient descent, Newton's method, and BFGS on given function. It plots the convergence path and loss function values for each method.

# Usage
At the top of [line_search.py](line_search.py), you must define the functions $f$, $df$, and $d2f$ which provide the value, first derivative, and second derivative of the function you wish to minimize. Then, you can run the script with the following command:

```python line_search.py```

For information on the command line arguments, run the following command:

```python line_search.py --help```