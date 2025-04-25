import numpy as np
import interpolatedFunction as fi
import matplotlib.pyplot as plt
import meanSquareAproximation as msa
from matplotlib.colors import ListedColormap
def avg_error(points,arr_nodes,w_x,m):
    return np.sqrt(np.sum([np.square((fi.f_x(x)-w_x(m,arr_nodes,x))) for x in points]))/len(points)

def max_error(points,arr_nodes,w_x,m):
    return max([abs(fi.f_x(x)-w_x(m,arr_nodes,x)) for x in points])

def print_plot(n_limit, m_limit):
    n_values = [i for i in range(2, n_limit + 1)]
    m_values = [i for i in range(2, m_limit + 1)]
    n_vals = []
    m_vals = []
    avg_err_vals = []
    max_err_vals = []
    x_values = np.linspace(-np.pi + 1, 2 * np.pi + 1, 1000)

    for n in n_values:
        for m in m_values:
            if n >= m:
                print(f"n:{n} and m:{m}")
                n_vals.append(n)
                m_vals.append(m)
                arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
                avg_err = avg_error(x_values, arr_nodes, msa.mean_square_aproximation, m)
                max_err = max_error(x_values, arr_nodes, msa.mean_square_aproximation, m)
                if np.isfinite(avg_err) and np.isfinite(max_err):
                    avg_err_vals.append(avg_err)
                    max_err_vals.append(max_err)

    # Colormap configuration to handle zero values
    cmap_avg = plt.cm.viridis
    cmap_avg_modified = ListedColormap(cmap_avg(np.linspace(0, 1, 256)))
    cmap_avg_modified.set_under('lightgray')  # Set color for values below the colormap's range (effectively for zero if the range starts above it)

    cmap_max = plt.cm.plasma
    cmap_max_modified = ListedColormap(cmap_max(np.linspace(0, 1, 256)))
    cmap_max_modified.set_under('lightgray')

    # Plotting Average Error
    plt.figure(figsize=(10, 8))
    heatmap_avg = plt.hist2d(n_vals, m_vals, weights=avg_err_vals, bins=[len(n_values), len(m_values)], cmap=cmap_avg_modified, vmin=1e-9) # Set a small vmin to exclude exact zero
    cbar_avg = plt.colorbar(heatmap_avg[3], label='Średni Błąd')
    plt.xlabel("Liczba węzłów (n)")
    plt.ylabel("Stopień aproksymacji (m)")
    plt.title(f"Gęstość Średniego Błędu (n >= m, n: 2-{n_limit}, m: 2-{m_limit})")
    plt.xticks(n_values)
    plt.yticks(m_values)
    plt.grid(True)
    plt.savefig(f"./plots/mean-square-aproximation-avg-error-density-n{n_limit}-m{m_limit}-ngeqm-nozero.png", format='png')
    plt.show()

    # Plotting Maximum Error
    plt.figure(figsize=(10, 8))
    heatmap_max = plt.hist2d(n_vals, m_vals, weights=max_err_vals, bins=[len(n_values), len(m_values)], cmap=cmap_max_modified, vmin=1e-9) # Set a small vmin
    cbar_max = plt.colorbar(heatmap_max[3], label='Maksymalny Błąd')
    plt.xlabel("Liczba węzłów (n)")
    plt.ylabel("Stopień aproksymacji (m)")
    plt.title(f"Gęstość Maksymalnego Błędu (n >= m, n: 2-{n_limit}, m: 2-{m_limit})")
    plt.xticks(n_values)
    plt.yticks(m_values)
    plt.grid(True)
    plt.savefig(f"./plots/mean-square-aproximation-max-error-density-n{n_limit}-m{m_limit}-ngeqm-nozero.png", format='png')
    plt.show()
print_plot(30,30)