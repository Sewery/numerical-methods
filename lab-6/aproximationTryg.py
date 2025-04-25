import aproximatedFunction as af
import numpy as np
import matplotlib.pyplot as plt

[a,b]=af.function_range()

def fourier(nodes_values_scaled,f_tryg,k):
    return 2/len(nodes_values_scaled)*(sum(
         af.f_x(x_i)*f_tryg(x_i_s*k) for [x_i,x_i_s] in nodes_values_scaled ))

def scaling(nodes_values):
    return [[x_i,((x_i-a)/(b-a))*2*np.pi -np.pi] for x_i in nodes_values]

def tryg_aproximation(nodes_values,x,m):
    
    nodes_values_scaled=scaling(nodes_values)

    a_0=fourier(nodes_values_scaled,np.cos,0)
    w_m_x=a_0/2

    for k in range(1,m+1):
        a_k=fourier(nodes_values_scaled,np.cos,k)
        b_k=fourier(nodes_values_scaled,np.sin,k)
        w_m_x+=a_k*np.cos(k*x)+b_k*np.sin(k*x)

    return w_m_x

def print_plot(m,n):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    af.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(a, b, n)
    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(a, b, 1000)
    y_values = np.array([tryg_aproximation(arr_nodes, x,m) for x in x_values])

    # Wartości węzłów
    data_x = np.array(arr_nodes)
    data_y = [af.f_x(x) for x in arr_nodes]

    # Narysuj wykres
    plt.plot(x_values, y_values, label="Wielomian aproksymujący", color='skyblue')
    plt.scatter(data_x, data_y, color='darkgreen', label="Węzły", zorder=3)

    # Oznacz wykresy
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Aproksymacja średniokwadratowa dla n={n} i m={m}")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/aprox-tryg-plot-{n}-{m}.png",format='png')
    plt.show()

def print_plots(arr_n_m):   
    for arr in arr_n_m:
        arr_m=arr[1]
        n=arr[0] 
        for m in arr_m:
            print_plot(m,n)

print_plots([[10,[2,3,4]],[21,[4,5,6,7,8]]])