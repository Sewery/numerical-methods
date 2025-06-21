import funkcjaInterpolowana as fi
import numpy as np
import matplotlib.pyplot as plt
def upper_bound(arr,x):
    left,right = 0,len(arr)
    while right-left!=0:
        curr=(right+left)//2
        if arr[curr]<x:
            left=curr+1
        elif arr[curr]>=x:
            right=curr
    return right
def clamped_quadratic_spline_value(x_nodes,x):
    i = upper_bound(x_nodes, x)
    if i != 0:
        i -= 1
    i = max(0, min(i, len(x_nodes)  - 2))
    c0 = fi.f_x(x_nodes[0])
    c1 = fi.f_x(x_nodes[1])
    h0 = x_nodes[1] - x_nodes[0]

    b = fi.df_x(x_nodes[0])
    a = (c1 - c0 - b * h0) / (h0**2)
    c = c0
    current_node_x = x_nodes[0] 

    for j in range(1, i + 1):
        hj_minus_1 = x_nodes[j] - x_nodes[j-1]
        next_b = 2 * a * hj_minus_1 + b
        cj = fi.f_x(x_nodes[j])
        cj_plus_1 = fi.f_x(x_nodes[j+1])
        hj = x_nodes[j+1] - x_nodes[j]
        next_a = (cj_plus_1 - cj - next_b * hj) / (hj**2)
        a = next_a
        b = next_b
        c = cj 
        current_node_x = x_nodes[j]
    dx = x - current_node_x
    return  a * dx**2 + b * dx + c
def print_plot(n):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = np.array([clamped_quadratic_spline_value(arr_nodes, x) for x in x_values])

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
    plt.title(f"Interpolacja zześciennych funkcji sklejanych z węzłami rozłożonymi równomiernie dla n={n}")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/spline-quadratic-evenly-plot-{n}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n)
#print_plots([2,4,10])
# print_plots([5,8])
# print_plots([11,20,30,34])
# print_plots([20,40,65,75])