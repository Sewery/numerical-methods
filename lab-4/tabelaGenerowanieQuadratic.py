import numpy as np
import pandas as pd
import funkcjaInterpolowana as fi
import naturalQuadraticSpline as nqs
import clampedQuadraticSpline as cqs
def avg_error(points,arr_nodes,w_x):
    return np.sqrt(np.sum([np.square((fi.f_x(x)-w_x(arr_nodes,x))) for x in points]))/len(points)

def max_error(points,arr_nodes,w_x):
    return max([abs(fi.f_x(x)-w_x(arr_nodes,x)) for x in points])

def export_error_table_to_csv(arr_n,filename):
    arr_avg_err_natrual_quadratic =[]
    arr_max_err_natrual_quadratic = []
    arr_avg_err_clamped_quadratic =[]
    arr_max_err_clamped_quadratic = []
    x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)

    for n in arr_n:
        arr_nodes_evenly = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)

        arr_avg_err_natrual_quadratic.append(f"{avg_error(x_values, arr_nodes_evenly, nqs.natural_quadratic_spline_value):.5e}")
        arr_max_err_natrual_quadratic.append(f"{max_error(x_values, arr_nodes_evenly, nqs.natural_quadratic_spline_value):.5e}")    
        arr_avg_err_clamped_quadratic.append(f"{avg_error(x_values, arr_nodes_evenly, cqs.clamped_quadratic_spline_value):.5e}")
        arr_max_err_clamped_quadratic.append(f"{max_error(x_values, arr_nodes_evenly, cqs.clamped_quadratic_spline_value):.5e}")   
                                    
    d = {
        "Liczba węzłów": pd.Series(arr_n, index=range(1,len(arr_n)+1), dtype=int),
        "Błąd średni.dla sklejania naturalnego": pd.Series(arr_avg_err_natrual_quadratic, index=range(1,  len(arr_n)+1)),
        "Błąd maks. dla sklejania naturalnego": pd.Series(arr_max_err_natrual_quadratic, index=range(1,  len(arr_n)+1)),
        "Błąd średni dla sklejania clamped": pd.Series(arr_avg_err_clamped_quadratic, index=range(1,  len(arr_n)+1)),
        "Błąd maks. dla sklejania clamped": pd.Series(arr_max_err_clamped_quadratic, index=range(1,  len(arr_n)+1))
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
export_error_table_to_csv([i for i in range(2,41,1)],"./data/quadratic-spline-out-2-35-1")
export_error_table_to_csv([i for i in range(40,61,5)],"./data/quadratic-spline-out-40-60-5")
export_error_table_to_csv([i for i in range(60,110,10)],"./data/quadratic-spline-out-60-100-10")