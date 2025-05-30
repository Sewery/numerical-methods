import numpy as np
import thomas as t
import gauss as g
import pandas as pd
import matplotlib.pyplot as plt
def generate_permutation_x(n):
    """Zwraca losową permutację n-elementową wektora złożonego z wartości 1 i -1."""
    return np.random.choice([-1, 1], n)
x_perms_arr = {}
for n in list(range(2, 21)) + [30, 50, 80, 100, 200,400]:
    x_perms_arr[n] = generate_permutation_x(n)

def calculate_b(A, x, dtype=np.float64):
    """ Obliczanie iloczynu macierzy A i wektora x (czyli wektor b)"""
    return  np.dot(A, x).astype(dtype)
def infinity_norm(A):
    """Norma jako suma wartości bezwzględnych wierszy macierzy A."""
    return np.max(np.sum(np.abs(A), axis=1))
def condition_number(A):
    """Oblicza współczynnik uwarunkowania macierzy A używając normy nieskończoność."""
    norm_A = infinity_norm(A)
    try:
        A_inv = np.linalg.inv(A)
        norm_A_inv = infinity_norm(A_inv)
    except np.linalg.LinAlgError:
        return np.inf  # Macierz A jest osobliwa i nie ma odwrotności
    return norm_A * norm_A_inv
def maximum_error(x_original, x_computed):
    """Oblicza maksymalny błąd między wektorem x_original a x_computed."""
    if x_computed is None:
        return np.nan  # lub np.inf, jeśli wolisz
    return np.max(np.abs(x_original - x_computed))

k=6 
m=3
def generate_abc_A_zad3(n, dtype=np.float64):
    a = np.zeros(n, dtype=dtype) 
    b = np.zeros(n, dtype=dtype) 
    c = np.zeros(n, dtype=dtype) 
    for i in range(n):
        b[i] = -m * (i+1) - k
        if i < n-1:
            c[i] = i+1
        if i > 0:
            a[i] = m / (i+1)
    return a, b, c

def generate_matrix_A_zad3(n, dtype=np.float64):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        A[i, i] = -m * (i+1) - k
        if i < n-1:
            A[i, i+1] = i+1
        if i > 0:
            A[i, i-1] = m / (i+1)
    return A
def zad_3_float(n):
    x_perm = x_perms_arr[n]
    a,b,c  = generate_abc_A_zad3(n, dtype=np.float32)
    A_float = generate_matrix_A_zad3(n, dtype=np.float32)
    x_float = x_perm.astype(np.float32)
    d_float = calculate_b(A_float, x_float, dtype=np.float32)
    solution_double_thomas,time1,max_mem1 = t.thomas_algorithm(a,b,c, d_float, dtype=np.float32)
    solution_double_gauss,time2,max_mem2 = g.gauss_elimination(A_float, d_float, dtype=np.float32)
    return [condition_number(A_float), maximum_error(x_perm, solution_double_thomas),maximum_error(x_perm, solution_double_gauss),time1,time2,max_mem1,max_mem2]
def zad_3_double(n):
    x_perm = x_perms_arr[n]
    a,b,c = generate_abc_A_zad3(n, dtype=np.float64)
    A_float = generate_matrix_A_zad3(n, dtype=np.float64)
    x_float = x_perm.astype(np.float64)
    d_float = calculate_b(A_float, x_float, dtype=np.float64)
    solution_double_thomas,time1,max_mem1 = t.thomas_algorithm(a,b,c, d_float,dtype=np.float64)
    solution_double_gauss,time2,max_mem2  = g.gauss_elimination(A_float, d_float,dtype=np.float64)
    return [condition_number(A_float), maximum_error(x_perm, solution_double_thomas),maximum_error(x_perm, solution_double_gauss),time1,time2,max_mem1,max_mem2]
def print_plot_errors(n_values, zad, filename):
    condition_values = []
    norma_thomas_values = []
    norma_gauss_values = []
    for n in n_values:
        values = zad(n)
        condition_values.append(f"{values[0]:.2e}")
        norma_thomas_values.append(values[1])
        norma_gauss_values.append(values[2])

    x_pos = range(len(n_values))  # indeksy zamiast wartości n

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, norma_thomas_values, marker='o', label="Norma maksimum dla metody Thomasa", color='lightgreen')
    plt.plot(x_pos, norma_gauss_values, marker='s', label="Norma maksimum dla metody Gaussa", color='blue')
    plt.xlabel("Rozmiar układu $n$", fontsize=12)
    plt.ylabel("Norma maksymalna", fontsize=12)
    plt.title("Porównanie norm maksymalnych dla metod Thomasa i Gaussa", fontsize=14)
    plt.yscale('log')
    plt.xticks(x_pos, [str(n) for n in n_values], rotation=45)  # równe odstępy, podpisy n
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', format='png')
    plt.show()
def print_plot_times(n_values, zad,filename):
    time_thomas_values = []
    time_gauss_values = []
    for n in n_values:
        values = zad(n)
        time_thomas_values.append(values[3])
        time_gauss_values.append(values[4])
    x_pos = range(len(n_values))

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, time_thomas_values, marker='o', label="Czas dla Thomasa", color='lightgreen')
    plt.plot(x_pos, time_gauss_values, marker='s', label="Czas dla Gaussa", color='blue')
    plt.xlabel("Rozmiar układu $n$", fontsize=12)
    plt.ylabel("Czas wykonania [s]", fontsize=12)
    plt.title("Porównanie czasów wykonania dla metod Thomasa i Gaussa", fontsize=14)
    plt.yscale('log')  # skala logarytmiczna dla osi Y
    # Podpisz każdy n na osi X
    plt.xticks(x_pos, [str(n) for n in n_values], rotation=45)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', format='png')
    plt.show()
def print_plot_memory(n_values, zad, filename):
    memory_thomas_values = []
    memory_gauss_values = []
    for n in n_values:
        values = zad(n)
        memory_thomas_values.append(values[5])
        memory_gauss_values.append(values[6])

    x_pos = range(len(n_values))  # indeksy zamiast wartości n

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, memory_thomas_values, marker='o', label="Maksymalna zużyta pamięć dla  metody Thomasa", color='lightgreen')
    plt.plot(x_pos, memory_gauss_values, marker='s', label="Maksymalna zużyta pamięć dla metody Gaussa", color='blue')
    plt.xlabel("Rozmiar układu $n$", fontsize=12)
    plt.ylabel("Zużycie pamięci [B]", fontsize=12)
    plt.title("Porównanie zużycia pamięci dla metod Thomasa i Gaussa", fontsize=14)
    plt.yscale('log')
    # Podpisz każdy n na osi X
    plt.xticks(x_pos, [str(n) for n in n_values], rotation=45)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', format='png')
    plt.show()
def export_error_table_to_csv(n_values,zad,filename):
    condition_values = []
    norma_thomas_values =[]
    norma_gauss_values =[]
    for n in n_values:
        values = zad(n)
        condition_values.append(f"{values[0]:.5e}")
        norma_thomas_values.append(f"{values[1]:.5e}")    
        norma_gauss_values.append(f"{values[2]:.5e}")                               
    d = {
        "n": pd.Series(n_values, index=range(1,len(n_values)+1), dtype=int),
        "Wartość uwarunkowania": pd.Series(condition_values, index=range(1,  len(n_values)+1)),
        "Wartość normy maksimum metodą Thomasa": pd.Series(norma_thomas_values, index=range(1,  len(n_values)+1)),
        "Wartość normy maksimum metodą Gaussa": pd.Series(norma_gauss_values, index=range(1,  len(n_values)+1)),
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
def export_times_table_to_csv(n_values,zad,filename):
    times_thomas_values = []
    times_gauss_values = []
    for n in n_values:
        values = zad(n)
        times_thomas_values.append(f"{values[3]:.2e}")    
        times_gauss_values.append(f"{values[4]:.2e}")                               
    d = {
        "n": pd.Series(n_values, index=range(1,len(n_values)+1), dtype=int),
        "Czas metodą Thomasa": pd.Series(times_thomas_values, index=range(1,  len(n_values)+1)),
        "Czas metodą Gaussa": pd.Series(times_gauss_values, index=range(1,  len(n_values)+1)),
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
def export_memory_table_to_csv(n_values,zad,filename):
    memory_thomas_values = []
    memory_gauss_values = []
    for n in n_values:
        values = zad(n)
        memory_thomas_values.append(f"{values[5]:.2e}")    
        memory_gauss_values.append(f"{values[6]:.2e}")                               
    d = {
        "n": pd.Series(n_values, index=range(1,len(n_values)+1), dtype=int),
        "Maksymalna zużyta pamięć metodą Thomasa": pd.Series(memory_thomas_values, index=range(1,  len(n_values)+1)),
        "Maksymalna zużyta pamięć Gaussa": pd.Series(memory_gauss_values, index=range(1,  len(n_values)+1)),
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
if __name__ == "__main__":
    export_error_table_to_csv(list(range(2, 21))+[30, 50], zad_3_float,"./data/error_table_float")
    export_error_table_to_csv(list(range(2, 21))+[30, 50, 80, 100, 200,400], zad_3_double, "./data/error_table_double")

    # export_times_table_to_csv(list(range(2, 21))+[30, 50], zad_3_float,"./data/times_table_float")
    # export_times_table_to_csv(list(range(2, 21))+[30, 50, 80, 100, 200,400], zad_3_double, "./data/times_table_double")

    # export_memory_table_to_csv(list(range(2, 21))+[30, 50], zad_3_float,"./data/memory_table_float")
    # export_memory_table_to_csv(list(range(2, 21))+[30, 50, 80, 100, 200,400], zad_3_double, "./data/memory_table_double")

    # print_plot_errors(list(range(2, 21))+[30, 50], zad_3_float,"./plots/errors-plot-float")
    # print_plot_errors(list(range(2, 21))+[30, 50, 80, 100, 200,400], zad_3_double,"./plots/errors-plot-double")

    # print_plot_times(list(range(2, 21))+[30, 50], zad_3_float,"./plots/times-plot-float")
    # print_plot_times(list(range(2, 21))+[30, 50, 80, 100, 200,400], zad_3_double,"./plots/times-plot-double")

    # print_plot_memory(list(range(2, 21))+[30, 50], zad_3_float,"./plots/memory-plot-float")
    # print_plot_memory(list(range(2, 21))+[30, 50, 80, 100, 200,400], zad_3_double,"./plots/memory-plot-double")