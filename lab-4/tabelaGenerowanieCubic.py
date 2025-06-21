import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import funkcjaInterpolowana as fi
import naturalCubicSpline as ncs
import clampedCubicSpline as ccs
def avg_error(points,arr_nodes,w_x):
    return np.sqrt(np.sum([np.square((fi.f_x(x)-w_x(arr_nodes,x))) for x in points]))/len(points)

def max_error(points,arr_nodes,w_x):
    return max([abs(fi.f_x(x)-w_x(arr_nodes,x)) for x in points])

def export_error_table_to_csv(arr_n,filename):
    arr_avg_err_natrual_cubic =[]
    arr_max_err_natrual_cubic = []
    arr_avg_err_clamped_cubic =[]
    arr_max_err_clamped_cubic = []
    x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)

    for n in arr_n:
        arr_nodes_evenly = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)

        arr_avg_err_natrual_cubic.append(f"{avg_error(x_values, arr_nodes_evenly, ncs.natural_cubic_spline_value):.5e}")
        arr_max_err_natrual_cubic.append(f"{max_error(x_values, arr_nodes_evenly, ncs.natural_cubic_spline_value):.5e}")    
        arr_avg_err_clamped_cubic.append(f"{avg_error(x_values, arr_nodes_evenly, ccs.clamped_cubic_spline_value):.5e}")
        arr_max_err_clamped_cubic.append(f"{max_error(x_values, arr_nodes_evenly, ccs.clamped_cubic_spline_value):.5e}")   
                                    
    d = {
        "Liczba węzłów": pd.Series(arr_n, index=range(1,len(arr_n)+1), dtype=int),
        "Błąd średni dla sklejania naturalnego": pd.Series(arr_avg_err_natrual_cubic, index=range(1,  len(arr_n)+1)),
        "Błąd maks. dla sklejania naturalnego": pd.Series(arr_max_err_natrual_cubic, index=range(1,  len(arr_n)+1)),
        "Błąd średni dla sklejania clamped": pd.Series(arr_avg_err_clamped_cubic, index=range(1,  len(arr_n)+1)),
        "Błąd maks. dla sklejania clamped": pd.Series(arr_max_err_clamped_cubic, index=range(1,  len(arr_n)+1))
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
export_error_table_to_csv([i for i in range(2,36,1)],"./data/cubic-spline-out-2-35-1")
export_error_table_to_csv([i for i in range(40,61,5)],"./data/cubic-spline-out-40-60-5")
export_error_table_to_csv([i for i in range(60,110,10)],"./data/cubic-spline-out-60-100-10")
