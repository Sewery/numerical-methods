import funkcjaInterpolowana as fi
# Funkcja interpolująca
def lagrange_d(arr_nodes, x, x_k):
    result = 1
    for node in arr_nodes:
        if node == x_k:
            continue
        result *= (x - node)
    return result

def lagrange_m(arr_nodes, x_k):
    result = 1
    for node in arr_nodes:
        if node == x_k:
            continue
        result *= (x_k - node)
    return result

def lagrange_interpolation(arr_nodes, x):
    result = 0
    for x_k in arr_nodes:
        l_k_x = lagrange_d(arr_nodes, x, x_k) / lagrange_m(arr_nodes, x_k)
        result += fi.f_x(x_k) * l_k_x
    return result