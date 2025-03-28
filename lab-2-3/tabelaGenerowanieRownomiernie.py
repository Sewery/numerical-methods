import numpy as np
import lagrange as l
import pandas as pd
import lagrange as l
import newton as nw
import funkcjaInterpolowana as fi

def avg_error(points,arr_nodes,w_x):
    return np.sqrt(np.sum([np.square((fi.f_x(x)-w_x(arr_nodes,x))) for x in points]))/len(points)
def max_error(points,arr_nodes,w_x):
    return max([abs(fi.f_x(x)-w_x(arr_nodes,x)) for x in points])
def export_error_table_to_csv(arr_n,filename):
    arr_avg_err_newton =[]
    arr_avg_err_lagrange =[]
    arr_max_err_newton = []
    arr_max_err_lagrange = []
    x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)
    for n in arr_n:
        arr_nodes_evenly = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)

        arr_max_err_newton.append(f"{max_error(x_values, arr_nodes_evenly, nw.newton_interpolation):.5e}")
        arr_max_err_lagrange.append(f"{max_error(x_values, arr_nodes_evenly, l.lagrange_interpolation):.5e}")  
        arr_avg_err_newton.append(f"{avg_error(x_values, arr_nodes_evenly, nw.newton_interpolation):.5e}")
        arr_avg_err_lagrange.append(f"{avg_error(x_values, arr_nodes_evenly, l.lagrange_interpolation):.5e}")                                    

    d = {
        "Liczba węzłów": pd.Series(arr_n, index=range(1,len(arr_n)+1), dtype=int),
        "Błąd maksymalny dla Newtona": pd.Series(arr_max_err_newton, index=range(1,  len(arr_n)+1)),
        "Błąd maksymalny dla Lagrange'a": pd.Series(arr_max_err_lagrange, index=range(1,  len(arr_n)+1)),
        "Błąd średni dla Newtona": pd.Series(arr_avg_err_newton, index=range(1,  len(arr_n)+1)),
        "Błąd średni dla Lagrange'a": pd.Series(arr_avg_err_lagrange, index=range(1,  len(arr_n)+1))
    }
    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)

#export_error_table_to_csv([i for i in range(2,36,1)],"evenly-out-2-35-1")
#export_error_table_to_csv([i for i in range(40,61,5)],"evenly-out-40-60-5")
export_error_table_to_csv([i for i in range(60,210,10)],"evenly-out-60-200-10")