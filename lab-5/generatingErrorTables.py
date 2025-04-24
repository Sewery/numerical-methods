import numpy as np
import pandas as pd
import interpolatedFunction as fi
import meanSquareAproximation as  msa
def avg_error(points,arr_nodes,w_x,m):
    return np.sqrt(np.sum([np.square((fi.f_x(x)-w_x(m,arr_nodes,x))) for x in points]))/len(points)

def max_error(points,arr_nodes,w_x,m):
    return max([abs(fi.f_x(x)-w_x(m,arr_nodes,x)) for x in points])

def export_error_table_to_csv(n_values,m_values,filename,error_f):
    data = []
    x_values = np.linspace(-np.pi + 1, 2 * np.pi + 1.1, 1000)
    columns = ["n"] + [f"m={m}" for m in m_values]

    for n in n_values:
        row = [n]
        for m in m_values:
            if n >= m:
                arr_nodes = np.linspace(-np.pi + 1, 2 * np.pi + 1, n)
                err = error_f(x_values, arr_nodes, msa.mean_square_aproximation, m)
                if np.isfinite(err):
                    row.append(f"{err:.3e}")
                else:
                    row.append('-')
            else:
                row.append('-')
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'{filename}.csv', index=False)
    return

export_error_table_to_csv([i for i in range(2,21)],[i for i in range(2,21)],"./data/msa-avg-err-data-2-20",avg_error)
export_error_table_to_csv([i for i in range(2,21)],[i for i in range(2,21)],"./data/msa-max-err-data-2-20",max_error)