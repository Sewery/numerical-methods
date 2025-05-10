import numpy as np
import matplotlib.pyplot as plt
import function as af

aprox_num = 0.5157485

def crit_residual(x_n, x_n_bef=0):
    return abs(af.f_x(x_n))

def crit_incremental(x_n, x_n_bef):
    return abs(x_n - x_n_bef)

def is_precise(x, aprox_num):
    # Porównanie do tylu miejsc po przecinku, co aprox_num
    decimal_places = len(str(aprox_num).split('.')[1])
    scale = 10 ** decimal_places
    x_floor = np.floor(x * scale) / scale
    aprox_floor = np.floor(aprox_num * scale) / scale
    return x_floor == aprox_floor

def metoda_siecznych(a, b, eps, f_x, criterion):
    x_0 = a
    x_1 = b
    x_n_bef = x_1
    x_n = (f_x(x_1) * x_0 - f_x(x_0) * x_1) / (f_x(x_1) - f_x(x_0))
    it = 0
    while eps < criterion(x_n, x_n_bef):
        temp = (f_x(x_n) * x_n_bef - f_x(x_n_bef) * x_n) / (f_x(x_n) - f_x(x_n_bef))
        x_n_bef = x_n
        x_n = temp
        it += 1
    return [x_n, it]

def metoda_newtona(x_0, eps, f_x, df_x, criterion):
    x_n_bef = x_0
    x_n = x_0 - (f_x(x_0) / df_x(x_0))
    it = 0
    while eps < criterion(x_n, x_n_bef):
        temp = x_n - (f_x(x_n) / df_x(x_n))
        x_n_bef = x_n
        x_n = temp
        it += 1
    return [x_n, it]

# sieczne
def sieczne_results(crit):
    print("Metoda siecznych")
    print("Zaczynamy od lewego brzegu")
    a, b = af.function_range()[0], af.function_range()[1]
    i = 0
    epsilons = [0.001, 0.0000001, 0.000000000001]
    a_values = np.arange(a, b, 0.1)
    for aa in a_values:
        print(f"a: {aa} b: {b}")
        for eps in epsilons:
            x_0_1, it_1 = metoda_siecznych(aa, b, eps, af.f_x, crit)
            print(f"eps: {eps}")
            print(f"x: {x_0_1:.8f} it: {it_1} is_precise: {is_precise(x_0_1, aprox_num)}")
        i += 1
    print("Zaczynamy od prawego brzegu")
    b_values = np.arange(b, a, -0.1)
    for bb in b_values:
        print(f"a: {a} b: {bb}")
        for eps in epsilons:
            x_0_1, it_1 = metoda_siecznych(a, bb, eps, af.f_x, crit)
            print(f"eps: {eps}")
            print(f"x: {x_0_1:.8f} it: {it_1} is_precise: {is_precise(x_0_1, aprox_num)}")
        i += 1

# newton
def newton_results(crit):
    print("Metoda newtona")
    a, b = af.function_range()[0], af.function_range()[1]
    epsilons = [0.001, 0.0000001, 0.000000000001]
    a_values = np.arange(a, b+0.1, 0.1)
    for x in a_values:
        print(f"Startowy x: {x}")
        for eps in epsilons:
            x_0_1, it_1 = metoda_newtona(x, eps, af.f_x, af.df_x, crit)
            print(f"eps: {eps}")
            print(f"x: {x_0_1:.8f} it: {it_1} is_precise: {is_precise(x_0_1, aprox_num)}")

# Kryterium residualne
#print("Kryterium residualne")
#sieczne_results(crit_residual)
#newton_results(crit_residual)

# Kryterium przyrostu
print("Kryterium przyrostu")
#sieczne_results(crit_incremental)
newton_results(crit_incremental)

def print_plot():
    # Wartości 1000 punktow na zadanym przedziale przed interpolacją funkcji
    af.draw()

    # Oznacz wykresy
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Funkcja f(x)")
    plt.grid()
    plt.legend()

    plt.savefig("./plots/function.png",format='png')
    plt.show()
# print_plot()