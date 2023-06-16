import numpy as np
import matplotlib.pyplot as plt

def return_arrays(x):
    a = x.keys()
    b = list(x.values())
    return np.array(list(a)),np.array(list(b))[:,0],np.array(list(b))[:,1]

def take_mean_dicts(x,runs):
    y = {}
    for var in x.values():
        for keys in var.keys():
            # keys = keys
            y[keys] = [0,0]
    for dict in x.values():
        for key,values in dict.items():
            y[key][0] += values[0]/runs
            y[key][1] += values[1]/runs
    return return_arrays(y)

def plot_graph(x,y,z,name,average = False,runs = 20):
    if average == True:
        x1,x2,x3 = take_mean_dicts(x,runs)
        y1,y2,y3 = take_mean_dicts(y,runs)
        z1,z2,z3 = take_mean_dicts(z,runs)
    else:
        x1,x2,x3 = return_arrays(x)
        y1,y2,y3 = return_arrays(y)
        z1,z2,z3 = return_arrays(z)
    
    figure, axis = plt.subplots(2,1)
    axis[0].plot(x1,x2,label = "Sigma Convg")
    axis[0].plot(y1,y2,label = "Conjugate Convergence")
    axis[0].plot(z1,z2,label = "Conjugate Sigma Convergence")
    axis[0].set_title("Sigma, Conjugate and Sigma Conjugate Sigma Convergence using L2 norm")
    axis[0].legend(loc = 'upper right')

    axis[1].plot(x1,x3,label = "Sigma Convg")
    axis[1].plot(y1,y3,label = "Conjugate Convergence")
    axis[1].plot(z1,z3,label = "Conjugate Sigma Convergence")
    axis[1].set_title("Sigma, Conjugate and Sigma Conjugate Sigma Convergence using Max norm")
    axis[1].legend(loc = 'upper right')

    figure.subplots_adjust(top = 2.5,right = 1.5)
    plt.savefig(name + ".png",bbox_inches='tight')
    return

def plot_graph_multiple(x,y,z,p,name,average = False,runs = 20):

    if average == True:
        x1,x2,x3 = take_mean_dicts(x,runs)
        y1,y2,y3 = take_mean_dicts(y,runs)
        z1,z2,z3 = take_mean_dicts(z,runs)
        p1,p2,p3 = take_mean_dicts(p,runs)
    else:
        x1,x2,x3 = return_arrays(x)
        y1,y2,y3 = return_arrays(y)
        z1,z2,z3 = return_arrays(z)
        p1,p2,p3 = return_arrays(p)
    
    figure, axis = plt.subplots(2,1)
    axis[0].plot(x1,x2,label = "Sigma Convg")
    axis[0].plot(y1,y2,label = "Jump Sigma Convg")
    axis[0].plot(z1,z2,label = "Conjugate Convergence")
    axis[0].plot(p1,p2,label = "Conjugate Sigma Convergence")
    axis[0].set_title("Sigma, Jump Sigma, Conjugate and Sigma Conjugate Sigma Convergence using L2 norm")
    axis[0].legend(loc = 'upper right')
    axis[0].set_xlim(-20,1000)

    axis[1].plot(x1,x3,label = "Sigma Convg")
    axis[1].plot(y1,y3,label = "Jump Sigma Convg")
    axis[1].plot(z1,z3,label = "Conjugate Convergence")
    axis[1].plot(p1,p3,label = "Conjugate Sigma Convergence")
    axis[1].set_title("Sigma, Jump Sigma, Conjugate and Sigma Conjugate Sigma Convergence using Max norm")
    axis[1].legend(loc = 'upper right')
    axis[1].set_xlim(-20,1000)

    figure.subplots_adjust(top = 2.5,right = 1.5)
    plt.savefig(name + ".png",bbox_inches='tight')
    return
