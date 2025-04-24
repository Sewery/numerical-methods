import numpy as np
import matplotlib.pyplot as plt

# Funkcja interpolowana
def f_x(x):
    return -2*x*np.sin(3*(x-1))
def df_x(x):
    return -6*x*np.cos(3 - 3*x) + 2*np.sin(3 - 3*x)
def draw_interpolated():
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = [f_x(x) for x in x_values]
    plt.plot(x_values, y_values, label="Funkcja aproksymująca", color='lightgreen')
def chebyshev_nodes(a,b,n):
    nodes = []
    for i in range(1,n+1):
        x_i = (1/2)*(a+b) +(1/2)*(b-a)*np.cos(((2*i-1)*np.pi)/(2*n))
        nodes.append(x_i)
    return nodes