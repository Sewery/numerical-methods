import numpy as np
import funkcjaInterpolowana as fi
import matplotlib.pyplot as plt

def interpolation_divided_diffs(arr_nodes):
    n = len(arr_nodes)
    divided_diffs = [[0] * n for _ in range(n)]
    for i in range(n):
        divided_diffs[i][0] = fi.f_x(arr_nodes[i])
    for j in range(1, n):
        for i in range(n - j):
            if arr_nodes[i + j] == arr_nodes[i]:
                divided_diffs[i][j] = fi.df_x(arr_nodes[i])
            else:
                divided_diffs[i][j] = (divided_diffs[i + 1][j - 1] - divided_diffs[i][j - 1]) / (arr_nodes[i + j] - arr_nodes[i])

    return [divided_diffs[0][j] for j in range(n)]

def hermite_interpolation(arr_nodes,x):
    n = len(arr_nodes)
    coeffs = interpolation_divided_diffs(arr_nodes)
    result = coeffs[0]
    nth_polynomal_val=1
    for i in range(1, n):
        nth_polynomal_val*=(x-arr_nodes[i-1])
        result += nth_polynomal_val*coeffs[i]
    return result