import matplotlib.pyplot as plt
import numpy as np
import newton as new
import lagrange as l
import funkcjaInterpolowana as fi
# Węzły Czebyszewa
def chebyshev_nodes(a,b,n):
    nodes = []
    for i in range(1,n+1):
        x_i = (1/2)*(a+b) +(1/2)*(b-a)*np.cos(((2*i-1)*np.pi)/(2*n))
        nodes.append(x_i)
    return nodes
def print_plot(n):

        arr_nodes = chebyshev_nodes(-np.pi + 1, 2 * np.pi + 1,n)
        # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
        x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
        y_values_l = np.array([l.lagrange_interpolation(arr_nodes, x) for x in x_values])
        y_values_n = np.array([new.newton_interpolation(arr_nodes, x) for x in x_values])

        # Wartości węzłów
        data_x = np.array(arr_nodes)
        data_y = [fi.f_x(x) for x in arr_nodes]

             # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
        fi.draw_interpolated()
        plt.plot(x_values, y_values_l, label="Wielomian interpolujący metodą Newtona", color='skyblue',linestyle='-', linewidth=3)
        plt.plot(x_values, y_values_n, label="Wielomian interpolujący metodą Lagrange'a", color='darkgreen',linestyle='--')
        # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
        plt.scatter(data_x, data_y, color='red', marker='o', label="Węzły" , zorder=2, s=30, edgecolors='black', linewidths=0.5)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Interpolacja z węzłami Czebyszewa dla n={n}")
        plt.grid()
        plt.legend(fontsize='small')

        plt.savefig(f"./plots/chebysev-plot-{n}.png",format='png')
        plt.show()
def print_plots(arr_n):    
    for n in arr_n:
         print_plot(n)
    
print_plots([4,8,11,15,30,65])