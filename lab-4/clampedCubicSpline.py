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
            
def clamped_spline_matrix_system(x_nodes):

    N = len(x_nodes) # Liczba węzłów (np. N=5 dla x_0..x_4)
    n = N - 1 # Liczba przedziałów (np. n=4 dla x_0..x_4)

    # Oblicz h_i = x_{i+1} - x_i dla i=0..n-1
    h = [x_nodes[i+1] - x_nodes[i] for i in range(n)]
    delta = [(fi.f_x(x_nodes[i+1]) - fi.f_x(x_nodes[i])) / h[i] for i in range(n)]

    matrix_a = np.zeros((N, N))
    matrix_b = np.zeros(N)

    matrix_a[0, 0] = 2 * h[0]
    matrix_a[0, 1] = h[0]
    matrix_b[0] = delta[0] - fi.df_x(x_nodes[0])

    for i in range(1, n): # pętla od i=1 do N-2
        matrix_a[i, i-1] = h[i-1]
        matrix_a[i, i] = 2 * (h[i-1] + h[i])
        matrix_a[i, i+1] = h[i]
        matrix_b[i] = delta[i] - delta[i-1]

    matrix_a[n, n-1] = h[n-1] 
    matrix_a[n, n] = 2 * h[n-1]
    matrix_b[n] = fi.df_x(x_nodes[n]) - delta[n-1] 

    sigma = np.linalg.solve(matrix_a, matrix_b)
    return h, sigma 

def clamped_cubic_spline_value(x_nodes, x):
 
    N = len(x_nodes) 
    n = N - 1
    h, sigma = clamped_spline_matrix_system(x_nodes)
    i =upper_bound(x_nodes,x)
    if i!=0: 
        i-=1
    i = max(0, min(i, n - 1))

    x_i = x_nodes[i]
    h_i = h[i]

    f_i = fi.f_x(x_i)
    f_i_1 = fi.f_x(x_nodes[i+1])

    b = (f_i_1 - f_i) / h_i - h_i * (sigma[i+1] + 2 * sigma[i])
    c = 3 * sigma[i]
    d = (sigma[i+1] - sigma[i]) / h_i

    dx = x - x_i
    return f_i + b * dx + c * dx**2 + d * dx**3

def print_plot(n):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = np.array([clamped_cubic_spline_value(arr_nodes, x) for x in x_values])

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
    plt.title(f"Interpolacja z warunkiem clamped boundary dla n={n}")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/cubic-clamped-plot-{n}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n)
#print_plots([2,4,10])
# print_plots([5,8])
# print_plots([11,15])
# print_plots([20,40,65,75])