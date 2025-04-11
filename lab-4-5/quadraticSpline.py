import funkcjaInterpolowana as fi
import numpy as np
import matplotlib.pyplot as plt
import clampedQuadraticSpline as cq
import naturalQuadraticSpline as nq
def print_plot(n):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji z brzegami naturalnymi
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    n_y_values = np.array([nq.natural_quadratic_spline_value(arr_nodes, x) for x in x_values])

    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji z brzegami naturalnymi
    c_y_values = np.array([cq.clamped_quadratic_spline_value(arr_nodes, x) for x in x_values])

    # Wartości węzłów
    data_x = np.array(arr_nodes)
    data_y = [fi.f_x(x) for x in arr_nodes]

    plt.plot(x_values, n_y_values, label="Clamped boundary sklejenie", color='skyblue')
    plt.plot(x_values, c_y_values, label="Naturalne sklejenie", color='magenta')
    plt.scatter(data_x, data_y, color='darkgreen', label="Węzły", zorder=3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Interpolacja funkcją sklejaną kwadratową dla n = {n}")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/quadratic-plot-{n}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n)
#print_plots([2,4,10])
print_plots([4,8,9,10,12,15])
print_plots([20,30,40,100])