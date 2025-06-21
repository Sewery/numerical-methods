import numpy as np
import matplotlib.pyplot as plt
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
     # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()

    arr_nodes = chebyshev_nodes(-np.pi + 1, 2 * np.pi + 1,n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = np.array([l.lagrange_interpolation(arr_nodes, x) for x in x_values])

    # Wartości węzłów
    data_x = np.array(arr_nodes)
    data_y = [fi.f_x(x) for x in arr_nodes]

    plt.plot(x_values, y_values, label="Wielomian interpolujący", color='skyblue')
    plt.scatter(data_x, data_y, color='darkgreen', label="Węzły Czebyszewa", zorder=3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Interpolacja Lagrange'a z węzłami Czebyszewa dla n={n}")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/chebysev/lagrange-chebysev-plot-{n}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n)
      
print_plots([5,8])
print_plots([11,15,20])
print_plots([40,100])