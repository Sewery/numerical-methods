import numpy as np
import math
print(2**(-5))
#Przed przekształceniem
# def formula_1_rek(x_0,k):
#     if k==0:
#         return x_0
#     prev=formula_1_rek(x_0,k-1)
#     return 2**(k)*(math.sqrt(1+2**(-k+1)*prev)-1)
# def formula_1_rek_np(x_0,k,dtype):
#     if k==0:
#         return np.array(x_0,dtype=dtype)
#     prev=formula_1_rek_np(x_0,k-1,dtype)
#     return np.power(2, k, dtype=dtype) * (
#         np.sqrt(1 + np.power(2, -k+1, dtype=dtype) * prev, dtype=dtype) - 1
#         )
# def formula_1_it(x_0,k):
#     x_k=x_0
#     for i in range(k):
#         x_k=2**(i+1)*(math.sqrt(1+2**(-i)*x_k)-1)
#     return x_k
#Po przekształceniu
# def formula_2_rek(x_0,k):
#     if k==0:
#         return x_0
#     prev=formula_2_rek(x_0,k-1)
#     return 2*prev/((math.sqrt(1+2**(-k+1)*prev)+1))
# def formula_2_rek_np(x_0,k,dtype):
#     if k==0:
#         return np.array(x_0,dtype=dtype)
#     prev=formula_2_rek_np(x_0,k-1,dtype)
#     return np.true_divide(2*prev,
#         np.sqrt(1 + np.power(2, -k+1, dtype=dtype) * prev, dtype=dtype) + 1,
#         dtype=dtype
#         )
# def formula_2_it(x_0,k):
#     x_k=x_0
#     for i in range(k):
#         x_k=2*x_k/((math.sqrt(1+2**(-i)*x_k)+1))
#     return x_k
def formula_1_it_np(x_0,k,dtype):
    x_k=np.array(x_0, dtype=dtype)
    for i in range(k):
       x_k = np.power(2, i+1, dtype=dtype) * (np.sqrt(1 + np.power(2, -i, dtype=dtype) * x_k, dtype=dtype) - 1)
    return x_k
def formula_2_it_np(x_0,k,dtype):
    x_k=np.array(x_0, dtype=dtype)
    for i in range(k):
        x_k=np.true_divide(2*x_k,
        np.sqrt(1 + np.power(2, -i, dtype=dtype) * x_k, dtype=dtype) + 1,
        dtype=dtype
        )
    return x_k
for x_0 in [5,10,20,100,10000]:
    for k in range(2,20):
        #float32
        print("For float32")
        n1 = formula_1_it_np(x_0,k,np.float32)
        n2 = formula_2_it_np(x_0,k,np.float32)
        print(f'formula 1 ={n1:.7f}')
        print(f'formula 2 ={n2:.7f}')
        n1 = formula_1_it_np(x_0,k,np.float64)
        n2 = formula_2_it_np(x_0,k,np.float64)
        print(f'formula 1 ={n1:.20f}')
        print(f'formula 2 ={n2:.20f}')
        
#         print(f'formula 2 ={n2:.20f}')
# for x_0 in [5,10,20,100,10000]:
#     for k in range(2,202):
#         #float64
#         print("For float64")
#         n1 = formula_1_it_np(x_0,k,np.float64)
#         n2 = formula_2_it_np(x_0,k,np.float64)
#         print(f'formula 1 ={n1:.20f}')
#         print(f'formula 2 ={n2:.20f}')
# x_0 = 5
# k = 11
# np.set_printoptions(precision=4)
# print(f'x_0= {x_0}')
# print(f'k= {k}')
# print(f'log(xo+1)= {np.log(x_0+1)}')
# print('#######################')
# print(f'formula 1 recursive = {formula_1_rek(x_0,k-1)}')
# print(f'formula 1 iterative = {formula_1_it(x_0,k)}')
# types = [[32,np.float32],[64,np.float64],[128,np.longdouble]]
# for x in types:
#     [prec,dtype]=x
#     print(f'formula 1 recursive using {dtype} ={formula_1_rek_np(x_0,k-1,dtype)}')
#     print(f'formula 1 itertive using {dtype} ={formula_1_it_np(x_0,k)}')
# print('#######################')
# print(f'formula 2 recursive = {repr(formula_2_rek(x_0,k-1))}')
# print(f'formula 2 iterative = {formula_2_it(x_0,k)}')
# for dtype in types:
#     [prec,dtype]=x
#     t1=formula_2_rek_np(x_0,k-1,dtype)
#     print(f'formula 2 recursive using {dtype} ={t1:.32f}')
#     print (f'formula 2 itertive using {dtype} ={formula_2_it_np(x_0,k)}')
# TODO blad wzgledny
#|(m-m_t)/m|
# blad