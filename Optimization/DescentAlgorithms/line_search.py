import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------

# def f(x,y):
#     return 8*x**2 + y**2

# def df(x,y):
#     return [16*x, 2*y]

# def d2f(x,y):
#     return [[16,0],[0,2]]

# ------------------------------------

# def f(x,y):
#     return (x-2)**2 + (x-4)**2 + (y-3)**2

# def df(x,y):
#     return [2*(x-2) + 2*(x-4), 2*(y-3)]

# def d2f(x,y):
#     return [[4,0],[0,2]]

# ------------------------------------

def f(x,y):
    # Some sinusoidal function
    return np.sin(x) + np.cos(y)

def df(x,y):
    return [np.cos(x), -np.sin(y)]

def d2f(x,y):
    return [[-np.sin(x), 0], [0, -np.cos(y)]]

# ------------------------------------

def main(args):
    # Perform backtracking line search to find step size
    def step_size(x, y, p):
        alpha = 1
        while f(x + alpha*p[0], y + alpha*p[1]) > f(x,y) + 0.7*alpha*np.dot(df(x,y),p):
            alpha = alpha*0.8
        return alpha


    def BFGS(x0, y0):
        x_i = x0
        y_i = y0
        num_iter = 0
        x_list = []
        y_list = []
        H = np.eye(2)
        grad_f = df(x_i,y_i)
        while np.linalg.norm(grad_f) > args.epsilon:
            x_list.append(x_i)
            y_list.append(y_i)
            # print(df(x_i,y_i))
            # print(H)
            d = -np.dot(H, df(x_i,y_i))
            alpha = step_size(x_i, y_i, d)
            x_i = x_i + alpha*d[0]
            y_i = y_i + alpha*d[1]
            s = np.array([x_i - x_list[-1], y_i - y_list[-1]])
            y = np.array(df(x_i,y_i)) - np.array(df(x_list[-1], y_list[-1]))
            p = 1/np.dot(y.T,s)
            H = H + (np.eye(2) - p*np.outer(s,y))@ H @(np.eye(2) - p*np.outer(y,s)) + p*np.outer(s,s)
            grad_f = df(x_i,y_i)
            num_iter += 1
        return x_list, y_list, num_iter

    def gradient_descent(x0, y0, alpha):
        x = x0
        y = y0
        x_list = []
        y_list = []
        num_iter = 0
        while np.linalg.norm(df(x,y)) > args.epsilon:
            x_list.append(x)
            y_list.append(y)
            x = x - alpha*df(x,y)[0]
            y = y - alpha*df(x,y)[1]
            num_iter += 1
        x_list.append(x)
        y_list.append(y)
        return x_list, y_list, num_iter


    def newton_method(x0, y0):
        x = x0
        y = y0
        x_list = []
        y_list = []
        num_iter = 0
        while np.linalg.norm(df(x,y)) > args.epsilon:
            x_list.append(x)
            y_list.append(y)
            x = x - np.linalg.inv(d2f(x,y))[0,0]*df(x,y)[0] - np.linalg.inv(d2f(x,y))[0,1]*df(x,y)[1]
            y = y - np.linalg.inv(d2f(x,y))[1,0]*df(x,y)[0] - np.linalg.inv(d2f(x,y))[1,1]*df(x,y)[1]
            num_iter += 1
        x_list.append(x)
        y_list.append(y)
        return x_list, y_list, num_iter
        
    def plot_contour_all(x_list_gd, y_list_gd, x_list_newton, y_list_newton, x_list_bfgs, y_list_bfgs, num_iter_gd, num_iter_newton, num_iter_bfgs):
        
        plot_vec = [1,1,1]
        
        x = np.linspace(1,14,1000)
        y = np.linspace(1,11,1000)
        X, Y = np.meshgrid(x,y)
        Z = f(X,Y)
        plt.contour(X,Y,Z,20)
        if(plot_vec[0]==1):
            plt.plot(x_list_gd, y_list_gd, color='green', label="Gradient Descent")
            plt.scatter(x_list_gd, y_list_gd, color='green', marker='x')
            
        if(plot_vec[1]==1):
            plt.plot(x_list_newton, y_list_newton, color='blue', label="Newton's Method")
            plt.scatter(x_list_newton, y_list_newton, color='blue', marker='x')
            
            
        if(plot_vec[2]==1):
            plt.plot(x_list_bfgs, y_list_bfgs, color='red', label="BFGS")
            plt.scatter(x_list_bfgs, y_list_bfgs, color='red', marker='x')
            
        plt.title("GD num_iters = " + str(num_iter_gd) + ", Newton: num_iters = " + str(num_iter_newton) + ", BFGS: num_iters = " + str(num_iter_bfgs))
        plt.legend()
        plt.savefig("plot1.png")
        plt.show()
        
        
    def plot_loss(x_list_gd, y_list_gd, x_list_newton, y_list_newton, x_list_bfgs, y_list_bfgs):
        loss_gd = []
        loss_newton = []
        loss_bfgs = []
        
        for i in range(len(x_list_gd)):
            loss_gd.append(np.sqrt((x_list_gd[i] - x_list_gd[-1])**2 + (y_list_gd[i] - y_list_gd[-1])**2))

        for i in range(len(x_list_newton)):
            loss_newton.append(np.sqrt((x_list_newton[i] - x_list_newton[-1])**2 + (y_list_newton[i] - y_list_newton[-1])**2))
        
        for i in range(len(x_list_bfgs)):
            loss_bfgs.append(np.sqrt((x_list_bfgs[i] - x_list_bfgs[-1])**2 + (y_list_bfgs[i] - y_list_bfgs[-1])**2))
            
        plt.plot(range(len(loss_gd)), loss_gd, color='green', label="Gradient Descent")
        plt.plot(range(len(loss_newton)), loss_newton, color='blue', label="Newton's Method")
        plt.plot(range(len(loss_bfgs)), loss_bfgs, color='red', label="BFGS")
        
        plt.legend()
        
        plt.title("Loss vs. Iterations")
        plt.xlim(0, 10)
        plt.savefig("plot_loss_1.png")
        
        plt.show()

    x_list_gd, y_list_gd, num_iter_gd = gradient_descent(args.x0, args.y0, args.lr)

    x_list_newton, y_list_newton, num_iter_newton = newton_method(args.x0, args.y0)

    x_list_bfgs, y_list_bfgs, num_iter_bfgs = BFGS(args.x0, args.y0)

    plot_contour_all(x_list_gd, y_list_gd, x_list_newton, y_list_newton, x_list_bfgs, y_list_bfgs, num_iter_gd, num_iter_newton, num_iter_bfgs)

    plot_loss(x_list_gd, y_list_gd, x_list_newton, y_list_newton, x_list_bfgs, y_list_bfgs)

    # plot_surface(x_list, y_list)

    # print(df(0,0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A program which compares the performance of gradient descent, Newton\'s method, and BFGS on given function.')
    parser.add_argument('--x0', type=float, default=4, help='Initial x value')
    parser.add_argument('--y0', type=float, default=5, help='Initial y value')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon value for convergence termination')
    args = parser.parse_args()
    
    main(args)
