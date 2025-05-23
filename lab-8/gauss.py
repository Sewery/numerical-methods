import numpy as np

def gauss_elimination(A, b):
    """
    Rozwiązuje układ równań liniowych Ax = b metodą eliminacji Gaussa.

    Parametry:
    A (np.array): Macierz współczynników (n x n).
    b (np.array): Wektor wyrazów wolnych (n x 1).

    Zwraca:
    np.array: Wektor rozwiązań x (n x 1) jeśli istnieje unikalne rozwiązanie.
    str: Komunikat o braku rozwiązania lub nieskończonej liczbie rozwiązań.
    """
    n = len(A)
    if A.shape[0] != A.shape[1]:
        return "Macierz A musi być kwadratowa."
    if A.shape[0] != len(b):
        return "Wymiary macierzy A i wektora b nie są zgodne."
    Ab = np.concatenate((A, b.reshape(-1, 1)), axis=1).astype(float)
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
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if Ab[i, i] == 0:
            if Ab[i, n] == 0:
                return "Układ ma nieskończenie wiele rozwiązań lub jest sprzeczny (zależne równania)."
            else:
                return "Układ nie ma rozwiązania (sprzeczny)."
        else:
            x[i] = Ab[i, n] / Ab[i, i]
            for j in range(i + 1, n):
                x[i] = x[i] - Ab[i, j] * x[j] / Ab[i, i]
    return x

