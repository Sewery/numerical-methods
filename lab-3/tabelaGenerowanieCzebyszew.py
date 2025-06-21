import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import funkcjaInterpolowana as fi
import hermite as h

def avg_error(points,arr_nodes,w_x):
    return np.sqrt(np.sum([np.square((fi.f_x(x)-w_x(arr_nodes,x))) for x in points]))/len(points)

def max_error(points,arr_nodes,w_x):
    return max([abs(fi.f_x(x)-w_x(arr_nodes,x)) for x in points])

def export_error_table_to_csv(arr_n,filename,mode):
    arr_avg_err_hermite =[]
    arr_max_err_hermite = []
    x_values = np.linspace(-np.pi+1, 2*np.pi+1.1, 1000)

    for n in arr_n:
        arr_nodes_chebysev = fi.chebyshev_nodes(-np.pi + 1, 2 * np.pi + 1,n)

        new_nodes = []
        if mode==2:
            for i in range(len(arr_nodes_chebysev)):
                new_nodes.append(arr_nodes_chebysev[i]) 
                if i % 2 == 0:
                    new_nodes.append(arr_nodes_chebysev[i])
            arr_nodes_chebysev = np.array(new_nodes)
        elif mode ==1:
            for i in range(len(arr_nodes_chebysev)):
                new_nodes.append(arr_nodes_chebysev[i]) 
                new_nodes.append(arr_nodes_chebysev[i])
            arr_nodes_chebysev = np.array(new_nodes)

        arr_max_err_hermite.append(f"{max_error(x_values, arr_nodes_chebysev, h.hermite_interpolation):.5e}")
        arr_avg_err_hermite.append(f"{avg_error(x_values, arr_nodes_chebysev, h.hermite_interpolation):.5e}")    
                                    
    d = {
        "Liczba węzłów": pd.Series(arr_n, index=range(1,len(arr_n)+1), dtype=int),
        "Błąd maksymalny dla Hermita": pd.Series(arr_max_err_hermite, index=range(1,  len(arr_n)+1)),
        "Błąd średni dla Hermita": pd.Series(arr_avg_err_hermite, index=range(1,  len(arr_n)+1))
    }

    df = pd.DataFrame(d)

    df.to_csv(f'{filename}.csv',index=False)
    return
export_error_table_to_csv([i for i in range(2,36,1)],"./data/czeb-hermite-out-2-35-1-co-drugi",2)
export_error_table_to_csv([i for i in range(2,36,1)],"./data/czeb-hermite-out-2-35-1-kazdy",1)
export_error_table_to_csv([i for i in range(40,61,5)],"./data/czeb-hermite-out-40-60-5-co-drugi",2)
export_error_table_to_csv([i for i in range(40,61,5)],"./data/czeb-hermite-out-40-60-5-kazdy",1)
export_error_table_to_csv([i for i in range(60,180,10)],"./data/czeb-hermite-out-60-180-10-co-drugi",2)
export_error_table_to_csv([i for i in range(60,180,10)],"./data/czeb-hermite-out-60-180-10-kazdy",1)
