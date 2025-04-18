import funkcjaInterpolowana as fi
import numpy as np
import matplotlib.pyplot as plt
import hermite as h
# n - liczba wezlow
# mode - 1 lub 2 (kazdy wezel lub co drugi drugiego stopnia)
def print_plot(n,mode):
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    fi.draw_interpolated()
    # Węzły
    arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
    arr_nodes_ch = fi.chebyshev_nodes(-np.pi + 1, 2 * np.pi + 1,n)
    # Tworzenie wezlow 2 stopnia
    # jesli mode = 1 to kazdy wezel 2 stopni
    # jesli mode = 2 to co drugi (pierwszy wezel jest 2 stopnia) wezel 2 stopnia
    new_nodes = []
    new_nodes_ch = []
    for i in range(len(arr_nodes)):
        new_nodes.append(arr_nodes[i]) 
        new_nodes.append(arr_nodes[i])
        new_nodes_ch.append(arr_nodes_ch[i])
        new_nodes_ch.append(arr_nodes_ch[i])


    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    # Wartosci y
    y_values = np.array([h.hermite_interpolation(new_nodes, x) for x in x_values])
    
    y_values_ch = np.array([h.hermite_interpolation(new_nodes_ch, x) for x in x_values])

    # Wartości węzłów
    data_x = np.array(arr_nodes)
    data_y = [fi.f_x(x) for x in arr_nodes]

    data_x_ch = np.array(arr_nodes_ch)
    data_y_ch= [fi.f_x(x) for x in arr_nodes_ch]

    plt.plot(x_values, y_values, label="Wielomian interpolujący z węzłami równomiernymi", color='skyblue',linestyle='-',linewidth=3)
    plt.plot(x_values, y_values_ch, label="Wielomian interpolujący z węzłami Czebyszewa", color='darkgreen',linestyle='--')
    #Rysowanie węzłów
    plt.scatter(data_x, data_y, color='red', marker='o', label="Węzły rozmieszone równomiernie" , zorder=2, s=30, edgecolors='black', linewidths=0.5)
    plt.scatter(data_x_ch, data_y_ch,color='black', marker='x', label="Węzły Czebyszewa", zorder=2, s=30, edgecolors='black')

    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)
    y_values = [fi.f_x(x) for x in x_values]

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Metoda interpolacji Hermita dla n={n}")
    plt.grid()
    plt.legend(fontsize='small')

    plt.savefig(f"./plots/hermite-plot-{n}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n,1)
print_plots([4,7,9,11,15,34])