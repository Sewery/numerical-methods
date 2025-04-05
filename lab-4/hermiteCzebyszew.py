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
    arr_nodes = fi.chebyshev_nodes(-np.pi + 1, 2 * np.pi + 1,n)
    # Tworzenie wezlow 2 stopnia
    # jesli mode = 1 to kazdy wezel 2 stopni
    # jesli mode = 2 to co drugi (pierwszy wezel jest 2 stopnia) wezel 2 stopnia
    new_nodes = []
    if mode==2:
        for i in range(len(arr_nodes)):
            new_nodes.append(arr_nodes[i]) 
            if i % 2 == 0:
                new_nodes.append(arr_nodes[i])
    elif mode ==1:
        for i in range(len(arr_nodes)):
            new_nodes.append(arr_nodes[i]) 
            new_nodes.append(arr_nodes[i])

    # Wartości 1000 punktow na zadanym przedziale po interpolacji funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1, 1000)
    # Wartosci y
    y_values = np.array([h.hermite_interpolation(new_nodes, x) for x in x_values])

    # Wartości węzłów
    data_x = np.array(arr_nodes)
    data_y = [fi.f_x(x) for x in arr_nodes]

    plt.plot(x_values, y_values, label="Wielomian interpolujący", color='skyblue')
    #Rysowanie węzłów
    if mode == 2:
        # Co drugi punkt - parzyste indeksy
        x_even = data_x[::2]
        y_even = data_y[::2]
        # Pozostałe punkty - nieparzyste indeksy
        x_odd = data_x[1::2]
        y_odd = data_y[1::2]
        # Rysowanie
        plt.scatter(x_even, y_even, color='darkgreen', label='Węzły parzyste', zorder=3)
        plt.scatter(x_odd, y_odd, color='orange', label='Węzły nieparzyste', zorder=3)
    elif mode==1:
        plt.scatter(data_x, data_y, color='darkgreen', label="Węzły", zorder=3)

    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)
    y_values = [fi.f_x(x) for x in x_values]

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Metoda Interpolacji Hermita dla n={n} oraz k={int(n/mode)+(1 if(mode==2 and n%2==1) else 0)} pochodnymi")
    plt.grid()
    plt.legend()

    plt.savefig(f"./plots/hermite-cheybsev-plot-{n}-{mode}.png",format='png')
    plt.show()
def print_plots(arr_n):   
    for n in arr_n:
        print_plot(n,1)
        print_plot(n,2)
print_plots([2,5,9])
print_plots([11,15])
print_plots([20,40,80])