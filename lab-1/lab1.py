from matplotlib.ticker import MultipleLocator
import numpy as np
import mpmath as mp
import pandas as pd
import matplotlib.pyplot as plt

def formula_1_it_np(x_0, k, dtype):
    x_k = np.array(x_0, dtype=dtype)
    for i in range(k):
        x_k = np.power(2, i + 1, dtype=dtype) * (
            np.sqrt(1 + np.power(2, -i, dtype=dtype) * x_k, dtype=dtype) - 1
        )
    return x_k


def formula_2_it_np(x_0, k, dtype):
    x_k = np.array(x_0, dtype=dtype)
    for i in range(k):
        x_k = np.true_divide(
            2 * x_k,
            np.sqrt(1 + np.power(2, -i, dtype=dtype) * x_k, dtype=dtype) + 1,
            dtype=dtype,
        )
    return x_k


def formula_1_it_mp(x_0, k):
    x_k = mp.mpf(x_0)
    for i in range(k):
        x_k = mp.mpf(2) ** mp.mpf(i + 1) * (
            mp.mpf(1 + (mp.mpf(2) ** mp.mpf(-i)) * x_k) ** mp.mpf(1 / 2) - 1
        )
    return x_k


def formula_2_it_mp(x_0, k):
    x_k = mp.mpf(x_0)
    for i in range(k):
        x_k = mp.mpf(
            mp.mpf(2 * x_k)
            / ((mp.mpf(1 + mp.mpf(2) ** (-i) * x_k) ** mp.mpf(1 / 2)) + 1),
        )
    return x_k
plt.rcParams.update({'font.size': 10})
# def draw_and_save_scatter(x_arr,y_arr,title,x_0,dtype,multiplicator_x=None,multiplicator_y=None):
#     pass

def float_charts_and_tables(x_0,k_range,multiplelocator_x=None,multiplelocator_y=None):
    format="svg"
    sum_series = np.log(x_0+1,dtype=np.float32)
    arr_float_a = []
    arr_float_b = []
    rel_err_a = []
    rel_err_b = []
    for k in k_range:
        a_x_k = formula_1_it_np(x_0, k, np.float32)
        b_x_k = formula_2_it_np(x_0, k, np.float32)
        arr_float_a.append(a_x_k)
        arr_float_b.append(b_x_k)
        rel_err_a.append(abs(np.true_divide(a_x_k-sum_series,sum_series,dtype= np.float32)))
        rel_err_b.append(abs(np.true_divide(b_x_k-sum_series,sum_series,dtype=np.float32)))
    ########
    # making table and saving it to svg file
    d = {
        "k": pd.Series(k_range, index=range(1,len(k_range)+1), dtype=int),
        "a_x_k": pd.Series(arr_float_a, index=range(1,  len(k_range)+1)),
        "rel_err_a": pd.Series(rel_err_a, index=range(1,  len(k_range)+1)),
        "b_x_k": pd.Series(arr_float_b, index=range(1,  len(k_range)+1)),
        "rel_err_b": pd.Series(rel_err_b, index=range(1,  len(k_range)+1))
    }
    df = pd.DataFrame(d)
    # format numeric columns
    formatted_values = []
    for i, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            if col == "k":
                formatted_row.append(f"{int(row[col])}")
            else:
                formatted_row.append(f"{row[col]:.7f}")
        formatted_values.append(formatted_row)

    _, ax = plt.subplots(figsize=(12, 14))
    ax.axis('off')
    table = ax.table(
        cellText=formatted_values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center',
        rowLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Adjust row height

    plt.savefig(f"table-a-float-{x_0}.{format}", dpi=150, bbox_inches='tight',format=format)
    ###########
    # making chart for a_x_k and saving it to svg file
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.scatter(k_range, pd.Series(arr_float_a, index=k_range),label="Ciąg a_x_k",s = [20*(1/2) for _ in k_range])
    ax.set_xlabel('k value')
    ax.set_ylabel('x_k value')
    ax.set_title(f'Wyniki dla a_x_k float, przy x_0 = {x_0}')

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if multiplelocator_y is not None:
        ax.yaxis.set_major_locator(multiplelocator_y)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    if multiplelocator_x is not None:
        ax.xaxisaxis.set_major_locator(multiplelocator_x)
    
    plt.savefig(f"plot-a-float-{x_0}.{format}",format=format)
    ###########
    # making chart for b_x_k and saving it to svg file
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.scatter(k_range, pd.Series(arr_float_b, index=k_range),label="Ciąg b_x_k",s = [20*(1/2) for _ in k_range])
    ax.set_xlabel('k value')
    ax.set_ylabel('x_k value')
    ax.set_title(f'Wyniki dla b_x_k dla float, przy x_0 = {x_0}')

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if multiplelocator_y is not None:
        ax.yaxis.set_major_locator(multiplelocator_y)
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if multiplelocator_x is not None:
        ax.xaxis.set_major_locator(multiplelocator_x)
    
    plt.savefig(f"plot-b-float-{x_0}.{format}",format=format)
def double_charts_and_tables(x_0,k_range,multiplelocator_x=None,multiplelocator_y_a=None,multiplelocator_y_b=None):
    format="svg"
    sum_series = np.log(x_0+1,dtype=np.float64)
    arr_float_a = []
    arr_float_b = []
    rel_err_a = []
    rel_err_b = []
    for k in k_range:
        a_x_k = formula_1_it_np(x_0, k, np.float64)
        b_x_k = formula_2_it_np(x_0, k, np.float64)
        arr_float_a.append(a_x_k)
        arr_float_b.append(b_x_k)
        rel_err_a.append(abs(np.true_divide(a_x_k-sum_series,sum_series,dtype=np.float64)))
        rel_err_b.append(abs(np.true_divide(b_x_k-sum_series,sum_series,dtype=np.float64)))
    ########
    # making table and saving it to svg file
    d = {
        "k": pd.Series(k_range, index=range(1,len(k_range)+1), dtype=int),
        "a_x_k": pd.Series(arr_float_a, index=range(1,  len(k_range)+1)),
        "rel_err_a": pd.Series(rel_err_a, index=range(1,  len(k_range)+1)),
        "b_x_k": pd.Series(arr_float_b, index=range(1,  len(k_range)+1)),
        "rel_err_b": pd.Series(rel_err_b, index=range(1,  len(k_range)+1))
    }
    df = pd.DataFrame(d)
    # format numeric columns
    formatted_values = []
    for i, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            if col == "k":
                formatted_row.append(f"{int(row[col])}")
            else:
                formatted_row.append(f"{row[col]:.17f}")
        formatted_values.append(formatted_row)

    _, ax = plt.subplots(figsize=(12, 14))
    ax.axis('off')
    table = ax.table(
        cellText=formatted_values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center',
        rowLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Adjust row height

    plt.savefig(f"table-a-double-{x_0}.{format}", dpi=150, bbox_inches='tight',format=format)
    ###########
    # making chart for a_x_k and saving it to svg file
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.scatter(k_range, pd.Series(arr_float_a, index=k_range),label="Ciąg a_x_k",s = [20*(1/2) for _ in k_range])
    ax.set_xlabel('k value')
    ax.set_ylabel('x_k value')
    ax.set_title(f'Wyniki a_x_k dla double, przy x_0 = {x_0}')

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if multiplelocator_y_a is not None:
        ax.yaxis.set_major_locator(multiplelocator_y_a)
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if multiplelocator_x is not None:
        ax.xaxis.set_major_locator(multiplelocator_x)
    plt.savefig(f"plot-a-double-{x_0}.{format}",format=format)
    ###########
    # making chart for b_x_k and saving it to svg file
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.scatter(k_range, pd.Series(arr_float_b, index=k_range),label="Ciąg b_x_k",s = [20*(1/2) for _ in k_range])
    ax.set_xlabel('k value')
    ax.set_ylabel('x_k value')
    ax.set_title(f'Wyniki b_x_k dla double, przy x_0 = {x_0} ,log(x_0+1)={np.log(x_0+1):.17f}')

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if multiplelocator_y_b is not None:
        ax.yaxis.set_major_locator(multiplelocator_y_b)
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if multiplelocator_x is not None:
        ax.xaxis.set_major_locator(multiplelocator_x)
    
    plt.savefig(f"plot-b-double-{x_0}.{format}",format=format)
def long_double_charts_and_tables(x_0,k_range,multiplelocator_x=None,multiplelocator_y_a=None,multiplelocator_y_b=None):
    format="svg"
    mp.mps = 36
    sum_series = mp.log(x_0+1)
    arr_float_a = []
    arr_float_b = []
    rel_err_a = []
    rel_err_b = []
    for k in k_range:
        a_x_k = formula_1_it_mp(x_0, k)
        b_x_k = formula_2_it_mp(x_0, k)
        arr_float_a.append(a_x_k)
        arr_float_b.append(b_x_k)
        rel_err_a.append(abs((a_x_k-sum_series)/sum_series))
        rel_err_b.append(abs((b_x_k-sum_series)/sum_series))
    # Convert mpf objects to Python floats for the DataFrame
    arr_float_a_py = [float(mp.nstr(x, n=33, min_fixed=-1, max_fixed=-1)) for x in arr_float_a]
    arr_float_b_py = [float(mp.nstr(x, n=33, min_fixed=-1, max_fixed=-1)) for x in arr_float_b]
    rel_err_a_py = [float(mp.nstr(x, n=33, min_fixed=-1, max_fixed=-1)) for x in rel_err_a]
    rel_err_b_py = [float(mp.nstr(x, n=33, min_fixed=-1, max_fixed=-1)) for x in rel_err_b]

    ########
    # making table and saving it to svg file
    d = {
        "k": pd.Series(k_range, index=range(1, len(k_range)+1), dtype=int),
        "a_x_k": pd.Series(arr_float_a_py, index=range(1, len(k_range)+1)),
        "rel_err_a": pd.Series(rel_err_a_py, index=range(1, len(k_range)+1)),
        "b_x_k": pd.Series(arr_float_b_py, index=range(1, len(k_range)+1)),
        "rel_err_b": pd.Series(rel_err_b_py, index=range(1, len(k_range)+1))
    }
    df = pd.DataFrame(d)
    # format numeric columns
    formatted_values = []
    for i, row in df.iterrows():
        formatted_row = []
        for col in df.columns:
            if col == "k":
                formatted_row.append(f"{int(row[col])}")
            else:
                formatted_row.append(f"{row[col]:.33f}")
        formatted_values.append(formatted_row)

    _, ax = plt.subplots(figsize=(20, 14))
    ax.axis('off')
    table = ax.table(
        cellText=formatted_values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center',
        rowLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Adjust row height

    plt.savefig(f"long-double-table-{x_0}.{format}", dpi=150, bbox_inches='tight',format=format)
    ###########
    # making chart for a_x_k and saving it to svg file
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.scatter(k_range, pd.Series(arr_float_a, index=k_range),label="Ciąg a_x_k",s = [20*(1/2) for _ in k_range])
    ax.set_xlabel('k value')
    ax.set_ylabel('x_k value')
    ax.set_title(f'Wyniki a_x_k dla long double, przy x_0 = {x_0}')

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if multiplelocator_y_a is not None:
        ax.yaxis.set_major_locator(multiplelocator_y_a)
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if multiplelocator_x is not None:
        ax.xaxis.set_major_locator(multiplelocator_x)
    plt.savefig(f"long-double-plot-{x_0}-a.{format}",format=format)
    ###########
    # making chart for b_x_k and saving it to svg file
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.scatter(k_range, pd.Series(arr_float_b, index=k_range),label="Ciąg b_x_k",s = [20*(1/2) for _ in k_range])
    ax.set_xlabel('k value')
    ax.set_ylabel('x_k value')
    ax.set_title(f'Wyniki b_x_k dla long double, przy x_0 = {x_0}')

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if multiplelocator_y_b is not None:
        ax.yaxis.set_major_locator(multiplelocator_y_b)
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if multiplelocator_x is not None:
        ax.xaxis.set_major_locator(multiplelocator_x)
    
    plt.savefig(f"long-double-plot-{x_0}-b.{format}",format=format)
## float
x_0=5
k_range=range(2,32)
float_charts_and_tables(x_0,k_range)
x_0=100000
k_range=range(10,40)
float_charts_and_tables(x_0,k_range,multiplelocator_y=MultipleLocator(0.2))
# ### double
x_0=5
k_range=range(2,62)
double_charts_and_tables(x_0,k_range,multiplelocator_x=MultipleLocator(2),multiplelocator_y_b=MultipleLocator(0.05))
x_0=100000
k_range=range(10,70)
double_charts_and_tables(x_0,k_range,multiplelocator_x=MultipleLocator(2),multiplelocator_y_a=MultipleLocator(0.2),multiplelocator_y_b=MultipleLocator(0.005))
# ## long double
x_0=5
k_range=range(2,62)
long_double_charts_and_tables(x_0,k_range,multiplelocator_x=MultipleLocator(2),multiplelocator_y_b=MultipleLocator(0.05))
x_0=100000
k_range=range(10,70)
long_double_charts_and_tables(x_0,k_range,multiplelocator_x=MultipleLocator(2),multiplelocator_y_a=MultipleLocator(0.2),multiplelocator_y_b=MultipleLocator(0.005))