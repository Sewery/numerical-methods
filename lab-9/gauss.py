import numpy as np
import timeit
import tracemalloc

def gauss_elimination(A, b, dtype=np.float64):
    n = len(A)
    start = timeit.default_timer()
    tracemalloc.start()
    A_array = np.array(A, dtype=dtype)
    b_array = np.array(b, dtype=dtype)

    Ab = np.concatenate((A_array, b_array.reshape(-1, 1)), axis=1)
    
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(Ab[k, i]) > abs(Ab[max_row, i]):
                max_row = k
        
        Ab[[i, max_row]] = Ab[[max_row, i]]

        if Ab[i, i] == 0:
            continue

        for k in range(i + 1, n):
            factor = Ab[k, i] / Ab[i, i]
            Ab[k, i:] = Ab[k, i:] - factor * Ab[i, i:]
    x = np.zeros(n, dtype=dtype)
    for i in range(n - 1, -1, -1):
        if Ab[i, i] == 0:
            if np.isclose(Ab[i, n], 0):
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                return None, timeit.default_timer()-start, peak
            else:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                return None, timeit.default_timer()-start, peak
        
        x[i] = Ab[i, n]
        for j in range(i + 1, n):
            x[i] -= Ab[i, j] * x[j]
        x[i] /= Ab[i, i]
    
    end = timeit.default_timer()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return x, end-start, peak 