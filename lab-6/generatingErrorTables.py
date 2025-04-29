import numpy as np
import pandas as pd
import aproximatedFunction as af
import aproximationTryg as at
def avg_error(points,arr_nodes,w_x,m):
    return np.sqrt(np.sum([np.square((af.f_x(x)-w_x(arr_nodes,x,m))) for x in points]))/len(points)

def max_error(points,arr_nodes,w_x,m):
    return max([abs(af.f_x(x)-w_x(arr_nodes,x,m)) for x in points])

def export_error_table_to_csv(n_values,m_values,filename,error_f):
    data = []
    x_values = np.linspace(-np.pi + 1, 2 * np.pi + 1.1, 1000)
    columns = ["m"] + [f"n={n}" for n in n_values]

    for m in m_values:
        row = [m]
        for n in n_values:
            if n >= 2*m+1:
                arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
                err = error_f(x_values, arr_nodes, at.tryg_aproximation, m)
                if np.isfinite(err):
                    row.append(f"{err:.2e}")
                else:
                    row.append('-')
            else:
                row.append('-')
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'{filename}.csv', index=False)
    return

export_error_table_to_csv([5,8,10,12,15,18,20,24,28,30,40,50],[2,3,4,5,6,7,8,9,10,11,13,14,19,24],"./data/msat-avg-err-data",avg_error)
export_error_table_to_csv([5,8,10,12,15,18,20,24,28,30,40,50],[2,3,4,5,6,7,8,9,10,11,13,14,19,24],"./data/msat-max-err-data",max_error)

