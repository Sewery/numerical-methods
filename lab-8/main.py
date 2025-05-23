import numpy as np
import pandas as pd
import gauss as g



# def zad_1():
#     for n in [2,5,7,8,10,12,14,15,20]:
#         print(f"\n--- n = {n} ---")
#         # Wygeneruj jedną permutację i użyj jej dla obu typów
#         x_perm = generate_permutation_x(n)
#         # float32 (single precision)
#         A_float = generate_matrix_A_zad1(n, dtype=np.float32)
#         x_float = x_perm.astype(np.float32)
#         b_float = calculate_b(A_float, x_float, dtype=np.float32)
#         print("float32:")
#         print(f"Macierz A:\n{A_float}")
#         print(f"Wektor x:\n{x_float}")
#         print(f"Wektor b:\n{b_float}")
#         solution_float = g.gauss_elimination(A_float, b_float)
#         print(f"Rozwiązanie x: {solution_float}")

#         # float64 (double precision)
#         A_double = generate_matrix_A_zad1(n, dtype=np.float64)
#         x_double = x_perm.astype(np.float64)
#         b_double = calculate_b(A_double, x_double, dtype=np.float64)
#         print("\ndouble (float64):")
#         print(f"Macierz A:\n{A_double}")
#         print(f"Wektor x:\n{x_double}")
#         print(f"Wektor b:\n{b_double}")
#         solution_double = g.gauss_elimination(A_double, b_double)
#         print(f"Rozwiązanie x: {solution_double}")
# def zad_2():
#     for n in [2,5,7,8,10,12,14,15,20,50,100,200]:
#         print(f"\n--- n = {n} ---")
#         # Wygeneruj jedną permutację i użyj jej dla obu typów
#         x_perm = generate_permutation_x(n)
#         # float32 (single precision)
#         A_float = generate_matrix_A_zad2(n, dtype=np.float32)
#         x_float = x_perm.astype(np.float32)
#         b_float = calculate_b(A_float, x_float, dtype=np.float32)
#         print("float32:")
#         print(f"Macierz A:\n{A_float}")
#         print(f"Wektor x:\n{x_float}")
#         print(f"Wektor b:\n{b_float}")
#         solution_float = g.gauss_elimination(A_float, b_float)
#         print(f"Rozwiązanie x: {solution_float}")

#         # float64 (double precision)
#         A_double = generate_matrix_A_zad2(n, dtype=np.float64)
#         x_double = x_perm.astype(np.float64)
#         b_double = calculate_b(A_double, x_double, dtype=np.float64)
#         print("\ndouble (float64):")
#         print(f"Macierz A:\n{A_double}")
#         print(f"Wektor x:\n{x_double}")
#         print(f"Wektor b:\n{b_double}")
#         solution_double = g.gauss_elimination(A_double, b_double)
#         print(f"Rozwiązanie x: {solution_double}")

def generate_matrix_A_zad1(n, dtype=np.float64):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            if i == 0:
                A[i, j] = 1
            else:
                A[i, j] = 1 / (i + 1 + j)
    return A
def generate_matrix_A_zad2(n, dtype=np.float64):
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i, j] = 2*(i+1) / (j+1)
            else:
                A[i, j] = A[j, i]
    return A
def generate_permutation_x(n):
    """Zwraca losową permutację n-elementową wektora złożonego z wartości 1 i -1."""
    return np.random.choice([-1, 1], n)
x_perms_arr = {}
for n in list(range(2, 21)) + [30, 50, 80, 100, 200]:
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

def zad_1_float(n):
    x_perm = x_perms_arr[n]
    A_float = generate_matrix_A_zad1(n, dtype=np.float32)
    x_float = x_perm.astype(np.float32)
    b_float = calculate_b(A_float, x_float, dtype=np.float32)
    solution_float = g.gauss_elimination(A_float, b_float)
    return [condition_number(A_float), maximum_error(x_perm, solution_float)]
def zad_1_double(n):
    x_perm = x_perms_arr[n]
    A_float = generate_matrix_A_zad1(n, dtype=np.float64)
    x_float = x_perm.astype(np.float64)
    b_float = calculate_b(A_float, x_float, dtype=np.float64)
    solution_double = g.gauss_elimination(A_float, b_float)
    return [condition_number(A_float), maximum_error(x_perm, solution_double)]
def zad_2_float(n):
    x_perm = x_perms_arr[n]
    A_float = generate_matrix_A_zad2(n, dtype=np.float32)
    x_float = x_perm.astype(np.float32)
    b_float = calculate_b(A_float, x_float, dtype=np.float32)
    solution_float = g.gauss_elimination(A_float, b_float)
    return [condition_number(A_float), maximum_error(x_perm, solution_float)]
def zad_2_double(n):
    """Zwraca condtion number i normę dla zadania 2"""
    x_perm = x_perms_arr[n]
    A_float = generate_matrix_A_zad2(n, dtype=np.float64)
    x_float = x_perm.astype(np.float64)
    b_float = calculate_b(A_float, x_float, dtype=np.float64)
    solution_double = g.gauss_elimination(A_float, b_float)
    return [condition_number(A_float), maximum_error(x_perm, solution_double)]
def zad_conditions_checks():
    for n in range(7,100):
        A1 = generate_matrix_A_zad1(n, dtype=np.float64)
        A2 = generate_matrix_A_zad2(n, dtype=np.float64)
        print(f"n={n} | cond(zad_1)={np.linalg.cond(A1):.2e} | cond(zad_2)={np.linalg.cond(A2):.2e}")
#zad_1()    
# zad_conditions_checks()
def export_error_table_to_csv(n_values,filename,zad):
    condition_values = []
    norma_values =[]

    for n in n_values:
        values = zad(n)
        condition_values.append(f"{values[0]:.5e}")
        norma_values.append(f"{values[1]:.5e}")    
    print(condition_values)                                
    d = {
        "n": pd.Series(n_values, index=range(1,len(n_values)+1), dtype=int),
        "Wartość uwarunkowania": pd.Series(condition_values, index=range(1,  len(n_values)+1)),
        "Wartość normy": pd.Series(norma_values, index=range(1,  len(n_values)+1)),
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
export_error_table_to_csv(list(range(2, 21)), "./data/zad_1_float", zad_1_float)
export_error_table_to_csv(list(range(2, 21)), "./data/zad_1_double", zad_1_double)
export_error_table_to_csv(list(range(2, 21)) + [30, 50, 80, 100, 200], "./data/zad_2_float", zad_2_float)
export_error_table_to_csv(list(range(2, 21)) + [30, 50, 80, 100, 200], "./data/zad_2_double", zad_2_double)