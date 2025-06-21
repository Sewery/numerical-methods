import numpy as np
import matplotlib.pyplot as plt
import newton as nw
import funkcjaInterpolowana as fi

def print_plot(n):
      # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
        fi.draw_interpolated()
        # Węzły
        arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
        # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
        x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
        y_values = np.array([nw.newton_interpolation(arr_nodes, x) for x in x_values])

        # Wartości węzłów
        data_x = np.array(arr_nodes)
        data_y = [fi.f_x(x) for x in arr_nodes]

        plt.plot(x_values, y_values, label="Wielomian interpolujący", color='skyblue')
        plt.scatter(data_x, data_y, color='darkgreen', label="Węzły", zorder=3)

        # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
        x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)
        y_values = [fi.f_x(x) for x in x_values]

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Interpolacja Newtona z węzłami rozłożonymi równomiernie dla n={n}")
        plt.grid()
        plt.legend()

        plt.savefig(f"newton-evenly-plot-{n}.png",format='png')
        plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n)
print_plots([2,4,10])
print_plots([5,8])
print_plots([11,15])
print_plots([20,40,65,75])