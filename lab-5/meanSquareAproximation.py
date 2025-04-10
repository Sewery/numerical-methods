import numpy as np
import funkcjaInterpolowana as fi
import matplotlib.pyplot as plt
def weight(x,i):
    return 1
def matrix_mean_square_aproximation(n,m,x_nodes):
    matrix_a = np.zeros((m+1, m+1))
    matrix_b = np.zeros(m+1)
    for j in range(m+1):
        matrix_b[j]=sum([weight(x_nodes[i],i)*fi.f_x(x_nodes[i])*(x_nodes[i]**(j)) for i in range(n)]) 
        for k in range(m+1):
            matrix_a[j,k]=sum([weight(x_nodes[i],i)*((x_nodes[i])**(k+j)) for i in range(n)]) 
    sol = np.linalg.solve(matrix_a,matrix_b)
    return sol
def mean_square_aproximation(m,x_nodes,x):
    a_matrix=matrix_mean_square_aproximation(len(x_nodes),m,x_nodes)
    return sum([a_matrix[j]*(x**j) for j in range(m+1)])
def print_plot(m,n):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = np.array([mean_square_aproximation(m,arr_nodes, x) for x in x_values])

    # Wartości węzłów
    data_x = np.array(arr_nodes)
    data_y = [fi.f_x(x) for x in arr_nodes]

    plt.plot(x_values, y_values, label="Wielomian interpolujący", color='skyblue')
    plt.scatter(data_x, data_y, color='darkgreen', label="Węzły", zorder=3)

    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = [fi.f_x(x) for x in x_values]

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Aproksymacja średniokwadratowa dla n={n} i m={m}")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/mean-square-aproximation-plot-{n}-{m}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(6,n)
print_plots([7,8])
print_plots([11,15])
# print_plots([20,40,65,75])
