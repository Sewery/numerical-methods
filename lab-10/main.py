import numpy as np
import matplotlib.pyplot as plt
import time
import csv

def generate_matrix(n, k=8, m=5):
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if j == i:
                A[i, i] = k
            else:
                A[i, j] = 1/(abs(i - j) + m)
    return A

x_perms_arr = {}
x_perms_arr_100 = {}
for n in list(range(2,1001)):
    x_perms_arr[n] = np.random.choice([-1, 1], n)
    x_perms_arr_100[n] = np.random.choice([-100, 100], n)

def jacobi_method(A, b, x0=None, tol=1e-10, max_iter=10000000, criterion=1):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64) if x0 is None else x0.copy().astype(np.float64)
    x_new = np.zeros(n, dtype=np.float64)
    tol = np.float64(tol) 
    
    diag = np.diag(A).astype(np.float64)
    diag_inv = np.float64(1.0) / diag
    
    A_no_diag = A.copy().astype(np.float64)
    np.fill_diagonal(A_no_diag, 0)
    
    if n < 300:
        warm_up_iters = 5
        for _ in range(warm_up_iters):
            x_temp = (b - A_no_diag @ x) * diag_inv
            x = x_temp.copy()
    
    start_time = time.time()
    iter_count = 0
    
    while iter_count < max_iter:
        x_new = (b - A_no_diag @ x) * diag_inv
        
        if criterion == 1:
            residual = np.linalg.norm(x_new - x, np.inf)
        elif criterion == 2:
            residual = np.linalg.norm(A @ x - b, np.inf)
            
        x = x_new.copy()
        iter_count += 1

        if residual < tol:
            break
            
    total_time = time.time() - start_time
    return x_new, iter_count, total_time/iter_count

def analyze_spectral_radius(A):
    D_inv = np.diag(1.0 / np.diag(A))
    R = A - np.diag(np.diag(A))
    iteration_matrix = -np.dot(D_inv, R)
    spectral_radius = max(abs(np.linalg.eigvals(iteration_matrix)))
    will_converge = spectral_radius < 1
    
    return spectral_radius, will_converge

def analyze_jacobi_performance(n_values, rho_values, max_iter=100000):
    results = {
        'n': n_values,
        'rho_values': rho_values,
        'spectral_radius': [],
        'will_converge': [],
        'iterations_crit1_zeros': {},
        'iterations_crit2_zeros': {},
        'time_iter_crit1_zeros': {},
        'time_iter_crit2_zeros': {},
        'error_crit1_zeros': {},
        'error_crit2_zeros': {},
        'iterations_crit1_100': {},
        'iterations_crit2_100': {},
        'time_iter_crit1_100': {},
        'time_iter_crit2_100': {},
        'error_crit1_100': {},
        'error_crit2_100': {}
    }
    
    for rho in rho_values:
        # Inicjalizacja list dla obu wektorów początkowych
        results['iterations_crit1_zeros'][rho] = []
        results['iterations_crit2_zeros'][rho] = []
        results['time_iter_crit1_zeros'][rho] = []
        results['time_iter_crit2_zeros'][rho] = []
        results['error_crit1_zeros'][rho] = []
        results['error_crit2_zeros'][rho] = []
        
        results['iterations_crit1_100'][rho] = []
        results['iterations_crit2_100'][rho] = []
        results['time_iter_crit1_100'][rho] = []
        results['time_iter_crit2_100'][rho] = []
        results['error_crit1_100'][rho] = []
        results['error_crit2_100'][rho] = []
    
    for n in n_values:
        print(f"Analizuje dla n={n}...")
        A = generate_matrix(n)
        b = x_perms_arr[n]
        exact_solution = np.linalg.solve(A, b)
        
        spectral_radius, will_converge = analyze_spectral_radius(A)
        results['spectral_radius'].append(spectral_radius)
        results['will_converge'].append(will_converge)
        
        for rho in rho_values:
            # Wektor początkowy zerowy
            x1_zeros, iter1_zeros, time_iter1_zeros = jacobi_method(A, b, x0=None, tol=rho, max_iter=max_iter, criterion=1)
            x2_zeros, iter2_zeros, time_iter2_zeros = jacobi_method(A, b, x0=None, tol=rho, max_iter=max_iter, criterion=2)
            
            error1_zeros = np.linalg.norm(x1_zeros - exact_solution, np.inf)
            error2_zeros = np.linalg.norm(x2_zeros - exact_solution, np.inf)
            
            results['iterations_crit1_zeros'][rho].append(iter1_zeros)
            results['iterations_crit2_zeros'][rho].append(iter2_zeros)
            results['time_iter_crit1_zeros'][rho].append(time_iter1_zeros)
            results['time_iter_crit2_zeros'][rho].append(time_iter2_zeros)
            results['error_crit1_zeros'][rho].append(error1_zeros)
            results['error_crit2_zeros'][rho].append(error2_zeros)
            
            # Wektor początkowy permutacja {-100, 100}
            x1_100, iter1_100, time_iter1_100 = jacobi_method(A, b, x0=x_perms_arr_100[n], tol=rho, max_iter=max_iter, criterion=1)
            x2_100, iter2_100, time_iter2_100 = jacobi_method(A, b, x0=x_perms_arr_100[n], tol=rho, max_iter=max_iter, criterion=2)
            
            error1_100 = np.linalg.norm(x1_100 - exact_solution, np.inf)
            error2_100 = np.linalg.norm(x2_100 - exact_solution, np.inf)
            
            results['iterations_crit1_100'][rho].append(iter1_100)
            results['iterations_crit2_100'][rho].append(iter2_100)
            results['time_iter_crit1_100'][rho].append(time_iter1_100)
            results['time_iter_crit2_100'][rho].append(time_iter2_100)
            results['error_crit1_100'][rho].append(error1_100)
            results['error_crit2_100'][rho].append(error2_100)
    
    return results

def analyze_spectral_radius_detailed(n_min=2, n_max=1000, step=1, k=8, m=5):
    n_values = list(range(n_min, n_max + 1, step))
    spectral_radii = []
    
    for n in n_values:
        A = generate_matrix(n, k, m)
        spec_rad, _ = analyze_spectral_radius(A)
        spectral_radii.append(spec_rad)
    
    first_non_convergent = None
    for i, radius in enumerate(spectral_radii):
        if radius >= 1:
            first_non_convergent = n_values[i]
            break
    
    return n_values, spectral_radii, first_non_convergent

def save_results_to_csv(results, output_dir):
    with open(f"{output_dir}/iterations_incremental_zeros.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['iterations_crit1_zeros'][rho][i]
                row.append(value if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/iterations_residual_zeros.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['iterations_crit2_zeros'][rho][i]
                row.append(value if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/iterations_incremental_100.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['iterations_crit1_100'][rho][i]
                row.append(value if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/iterations_residual_100.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['iterations_crit2_100'][rho][i]
                row.append(value if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/time_iter_incremental_zeros.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['time_iter_crit1_zeros'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/time_iter_residual_zeros.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['time_iter_crit2_zeros'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/time_iter_incremental_100.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['time_iter_crit1_100'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/time_iter_residual_100.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['time_iter_crit2_100'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/errors_incremental_zeros.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['error_crit1_zeros'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/errors_residual_zeros.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['error_crit2_zeros'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/errors_incremental_100.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['error_crit1_100'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/errors_residual_100.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['n'] + [f'{rho:.1e}' for rho in results['rho_values']]
        writer.writerow(['Dokladnosc rho'])
        writer.writerow(header)
        
        for i, n in enumerate(results['n']):
            row = [n]
            for rho in results['rho_values']:
                value = results['error_crit2_100'][rho][i]
                row.append(f'{value:.6e}' if value is not None else 'N/A')
            writer.writerow(row)
    
    with open(f"{output_dir}/spectral_radius.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'promien spektralny'])
        
        for i, n in enumerate(results['n']):
            writer.writerow([n, f'{results["spectral_radius"][i]:.6e}'])

def create_plots(detailed_spectral_data, output_dir):
    n_values_detailed, spectral_radii, first_non_convergent = detailed_spectral_data
    
    plt.figure(figsize=(12, 8))
    plt.plot(n_values_detailed, spectral_radii, 'b-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Granica zbieżności')
    
    if first_non_convergent:
        plt.axvline(x=first_non_convergent, color='g', linestyle='--', 
                    label=f'Pierwszy niezbieżny n={first_non_convergent}')
    
    plt.grid(True)
    plt.xlabel('Rozmiar macierzy (n)')
    plt.ylabel('Promień spektralny')
    plt.title('Promień spektralny macierzy iteracji w zależności od rozmiaru macierzy')
    plt.legend()
    plt.savefig(f"{output_dir}/spectral_radius.png")
    plt.close()

def main():
    n_values = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 739]
    rho_values = [1e-3, 1e-7, 1e-15]
    
    max_iter = 100000
    
    output_dir_csv = "results/csv"
    output_dir_plots = "results/plots"
    
    print("Analiza promienia spektralnego...")
    n_values_detailed, spectral_radii, first_non_convergent = analyze_spectral_radius_detailed()
    
    if first_non_convergent:
        print(f"Metoda Jacobiego przestaje zbiegać dla n >= {first_non_convergent}")
    else:
        print("Metoda Jacobiego zbiegała dla wszystkich testowanych n")
    
    print("Analiza wydajności dla dwóch typów wektorów początkowych...")
    valid_n_values = [n for n in n_values if n < first_non_convergent] if first_non_convergent else n_values
    if not valid_n_values:
        print("Brak wartości n, dla których metoda zbiegnie. Kończę analizę.")
        return
    
    results = analyze_jacobi_performance(valid_n_values, rho_values, max_iter)
    
    save_results_to_csv(results, output_dir_csv)
    
    create_plots((n_values_detailed, spectral_radii, first_non_convergent), output_dir_plots)
    

if __name__ == "__main__":
    main()