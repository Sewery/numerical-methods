import numpy as np
import thomas as t
import gauss as g
import pandas as pd
import matplotlib.pyplot as plt
import timeit
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
    return np.max(np.abs(x_original - x_computed))

# k=6 , m=3
def generate_abc_A_zad3(n, dtype=np.float64):
    a = np.array([-3*(i+1) -6 for i in range(n)],dtype=dtype)
    b = np.array([(i+1) for i in range(n)],dtype=dtype)
    c = np.array([3/(i+1) for i in range(n)],dtype=dtype)
    return a,b,c

def generate_matrix_A_zad3(n, dtype=np.float64):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i, j] = -3*(i+1) -6 
            elif i == j+1:
                 A[i, j] = (i+1)
            elif i>0 and i == j-1:
                A[i, j] = 3/(i+1)
    return A
def zad_3_float(n):
    x_perm = x_perms_arr[n]
    a,b,c  = generate_abc_A_zad3(n, dtype=np.float32)
    A_float = generate_matrix_A_zad3(n, dtype=np.float32)
    x_float = x_perm.astype(np.float32)
    d_float = calculate_b(A_float, x_float, dtype=np.float32)
    solution_double_thomas,time1 =  t.thomas_algorithm(a,b,c, d_float)
    solution_double_gauss,time2 = g.gauss_elimination(A_float, d_float)
    return [condition_number(A_float), maximum_error(x_perm, solution_double_thomas),maximum_error(x_perm, solution_double_gauss)]
def zad_3_double(n):
    x_perm = x_perms_arr[n]
    a,b,c = generate_abc_A_zad3(n, dtype=np.float64)
    A_float = generate_matrix_A_zad3(n, dtype=np.float64)
    x_float = x_perm.astype(np.float64)
    d_float = calculate_b(A_float, x_float, dtype=np.float64)
    solution_double_thomas,time1 = t.thomas_algorithm(a,b,c, d_float)
    solution_double_gauss,time2  = g.gauss_elimination(A_float, d_float)
    return [condition_number(A_float), maximum_error(x_perm, solution_double_thomas),maximum_error(x_perm, solution_double_gauss)]
def print_plot_errors(n_values, zad, filename):
    pass
def print_plot_times(n_values, zad, filename):
    pass
def print_plot_memory(n_values, zad, filename):
    pass
def export_error_table_to_csv(n_values,filename,zad):
    condition_values = []
    norma_thomas_values =[]
    norma_gauss_values =[]
    for n in n_values:
        values = zad(n)
        condition_values.append(f"{values[0]:.5e}")
        norma_thomas_values.append(f"{values[1]:.5e}")    
        norma_gauss_values.append(f"{values[2]:.5e}")
    print(condition_values)                                
    d = {
        "n": pd.Series(n_values, index=range(1,len(n_values)+1), dtype=int),
        "Wartość uwarunkowania": pd.Series(condition_values, index=range(1,  len(n_values)+1)),
        "Wartość normy metodą Thomasa": pd.Series(norma_thomas_values, index=range(1,  len(n_values)+1)),
        "Wartość normy metodą Gaussa": pd.Series(norma_gauss_values, index=range(1,  len(n_values)+1)),
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
export_error_table_to_csv(list(range(2, 21))+[30, 50, 80, 100, 200,400], "./data/zad_3_float", zad_3_float)
export_error_table_to_csv(list(range(2, 21))+[30, 50, 80, 100, 200,400], "./data/zad_3_double", zad_3_double)
