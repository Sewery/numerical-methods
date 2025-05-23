import timeit
def thomas_algorithm(a, b, c, d):
    start = timeit.timeit()
    """
    Złożoność O(n)
    Rozwiązuje układ równań liniowych z macierzą trójdiagonalną Ax = d
    """
    n = len(d)
    # Kopie, żeby nie nadpisywać oryginałów
    ac, bc, cc, dc = a[:], b[:], c[:], d[:]
    # Modyfikacja współczynników
    for i in range(1, n):
        m = ac[i] / bc[i-1]
        bc[i] = bc[i] - m * cc[i-1]
        dc[i] = dc[i] - m * dc[i-1]
    # Podstawienie wsteczne
    x = [0] * n
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    end = timeit.timeit()
    return x,end-start
