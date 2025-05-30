import timeit
import tracemalloc
import numpy as np

def thomas_algorithm(a, b, c, d, dtype=np.float64):
    n = len(d)
    
    # Pomiar czasu i pamięci
    start = timeit.default_timer()
    tracemalloc.start()
    
    # Konwersja do odpowiedniego typu danych używając NumPy
    ac = np.array(a, dtype=dtype)
    bc = np.array(b, dtype=dtype)
    cc = np.array(c, dtype=dtype)
    dc = np.array(d, dtype=dtype)
    
    # Modyfikacja współczynników
    for i in range(1, n):
        m = ac[i] / bc[i-1]
        bc[i] = bc[i] - m * cc[i-1]
        dc[i] = dc[i] - m * dc[i-1]
    
    # Podstawienie wsteczne
    x = np.zeros(n, dtype=dtype)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    
    # Pomiar czasu i pamięci
    end = timeit.default_timer()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return x, end-start, peak
