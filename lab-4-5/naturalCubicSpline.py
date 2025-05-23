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
            
def matrix_of_equations(x_nodes):
    n=len(x_nodes)
    h= [x_nodes[i+1]-x_nodes[i]for i in range(n-1)]
    delta = [(fi.f_x(x_nodes[i+1])-fi.f_x(x_nodes[i]))/h[i] for i in range(n-1)]
    matrix_a = np.zeros((n, n))
    matrix_b = np.zeros(n)
    for i in range(1,n-1):
        matrix_a[i][i-1]=h[i-1]
        matrix_a[i][i]=2*(h[i-1]+h[i])
        matrix_a[i][i+1]=h[i]
        matrix_b[i] = delta[i] - delta[i-1]
    matrix_a[0][0] = 1
    matrix_a[n-1][n-1] = 1
    sigma = np.linalg.solve(matrix_a,matrix_b)
    return h,sigma
def natural_cubic_spline_value(x_nodes,x):

    h,sigma=matrix_of_equations(x_nodes)
    i =upper_bound(x_nodes,x)
    if i!=0: 
        i-=1
    i = max(0, min(i, len(x_nodes)  - 2))
    x_i= x_nodes[i]
    x_i_1= x_nodes[i+1]
    h_i = h[i]
    f_i = fi.f_x(x_i)
    f_i_1 = fi.f_x(x_i_1)

    b= (f_i_1 - f_i)/h[i] - h_i*(sigma[i+1]+2*sigma[i])
    c=3*sigma[i]
    d=(sigma[i+1]-sigma[i])/h[i]
    return f_i+b*(x-x_i)+c*(x-x_i)**2+d*(x-x_i)**3
def print_plot(n):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    y_values = np.array([natural_cubic_spline_value(arr_nodes, x) for x in x_values])

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
    plt.title(f"Interpolacja z naturalnym warunkiem brzegowym dla n={n}")
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