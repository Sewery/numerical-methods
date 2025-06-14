import numpy as np
import matplotlib.pyplot as plt

x_0=-np.pi/4
x_1=2*np.pi
m=3
k=2

def dokladne_y(x):
    return np.exp(-k*np.cos(m*x)) - k*np.cos(m*x)+1
# y(x_0) = a 
y_0 = a = dokladne_y(x_0)
def f(x,y):
    '''
    Równanine to
    du/dt + f(u,t) = 0  u = u(t) 
    u nas to:
    y' + f(x,y) = y' - k*m*y*np.sin(m*x) - (k^2)*m*sin(m*x)*cos(m*x) = 0
    tak więc f to
    '''
    return k*m*y*np.sin(m*x) +k**2*m*np.sin(m*x)*np.cos(m*x)
###
# Algorytm Eulera
# u^(n+1) = u^n - f(u^n,t^n)*delta t
# y(x_n+1) = y(x_n) - f(y(x_n),x_n)*delta x
#
# Krok x gwartujący stabilność
# delta x <= 2/(df/dx|n)
###
def euler(h):
    n = int((x_1 - x_0) / h)
    x = [0]*(n+1)
    y = [0]*(n+1)
    x[0] = x_0
    y[0] = y_0
    
    for i in range(n):
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h * f(x[i], y[i])
    
    return x, y

def pochodna_f(x):
    '''
    Pochodna funkcji f(y,x) względem y
    Dla warunku stabliności >0
    '''
    return k*m*np.cos(m*x) 
###
#4 punktowa metoda Rungego-Kutty
# y(x_n+1) = y(x_n) + h/6 * (k1 + 2*k2 + 2*k3 + k4)
# ro_4(h) = O(h^4) to znaczy, że błąd spada wraz ze wzrostem h do czwartej potęgi
##
def runge_kutta_4(h):
    n = int((x_1 - x_0) / h)
    x = [0] * (n+1)
    y = [0] * (n+1)
    x[0] = x_0
    y[0] = y_0
    
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2*k1)
        k3 = f(x[i] + h/2, y[i] + h/2*k2)
        k4 = f(x[i] + h, y[i] + h*k3)
        
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return x, y

h_values = [0.1, 0.01,0.0001]

for idx, h in enumerate(h_values):
    plt.figure(figsize=(12, 8))
    x_euler, y_euler = euler(h)
    
    x_rk4, y_rk4 = runge_kutta_4(h)
    
    x_exact = np.linspace(x_0, x_1, 1000)
    y_exact = dokladne_y(x_exact)
    
    # Obliczanie błędu maksymalnego
    euler_errors = [abs(y_euler[i] - dokladne_y(x_euler[i])) for i in range(len(x_euler))]
    rk4_errors = [abs(y_rk4[i] - dokladne_y(x_rk4[i])) for i in range(len(x_rk4))]
    
    max_euler_error = max(euler_errors)
    max_rk4_error = max(rk4_errors)
    
    print(f"h = {h}:")
    print(f"Maksymalny błąd Eulera: {max_euler_error:.8f}")
    print(f"Maksymalny błąd Rungego-Kutty: {max_rk4_error:.8f}")
    print()
    
    # Wykresy
    plt.plot(x_exact, y_exact, 'k-', label='Rozwiązanie dokładne')
    plt.plot(x_euler, y_euler, 'b--', label=f'Euler h={h}')
    plt.plot(x_rk4, y_rk4, 'r:', label=f'Runge kutta 4 h={h}')
    plt.title(f'Porównanie dla kroku h = {h}')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'porownanie-krok-{idx}.png')
    plt.show()
