import numpy as np
import matplotlib.pyplot as plt

# Przedzialy
def function_range():
    return [-0.7,1.1]

# Funkcja interpolowana
def f_x(x):
    return (x-1)*np.exp(-14*x)+x**12

# Pochodna funkcji interpolowanej
def df_x(x):
    return np.exp(-14*x)*(15-14*x)+12*x**11

def draw():
    x_values = np.linspace(-0.7, 1.2, 1000)
    y_values = [f_x(x) for x in x_values]
    plt.plot(x_values, y_values, color='lightgreen')