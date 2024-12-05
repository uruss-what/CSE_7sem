import numpy as np
import matplotlib.pyplot as plt

# формула для обновления значений: yi+1 = yi + h * df(xi,yi)
def euler_method(f, y0, x_range, h):
    x_values = np.arange(x_range[0], x_range[1] + h, h)
    y_values = np.zeros_like(x_values)
    y_values[0] = y0
    for i in range(1, len(x_values)):
        y_values[i] = y_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1]) #формула эйлера
    return x_values, y_values

def func_a(x, y):
    return 0.5 * y

def func_b(x, y):
    return 2 * x + 3 * y

def system_c(x1, x2, h, t_range): #гармоническое колебание (маятник)
    t_values = np.arange(t_range[0], t_range[1] + h, h)
    x1_values = np.zeros_like(t_values)
    x2_values = np.zeros_like(t_values)
    x1_values[0] = 1
    x2_values[0] = 0
    for i in range(1, len(t_values)): # для каждой х1 х2 метод отдельно
        x1_values[i] = x1_values[i - 1] + h * x2_values[i - 1]
        x2_values[i] = x2_values[i - 1] - h * x1_values[i - 1]
    return t_values, x1_values, x2_values

def system_d(x1, x2, h, t_range): # экспоненц тк коэф х1 =4
    t_values = np.arange(t_range[0], t_range[1] + h, h)
    x1_values = np.zeros_like(t_values)
    x2_values = np.zeros_like(t_values)
    x1_values[0] = 1
    x2_values[0] = 1
    for i in range(1, len(t_values)):
        x1_values[i] = x1_values[i - 1] + h * x2_values[i - 1]
        x2_values[i] = x2_values[i - 1] + h * 4 * x1_values[i - 1]
    return t_values, x1_values, x2_values

h = 0.025
x_range = [0, 10]

x_a, y_a = euler_method(func_a, y0=1, x_range=x_range, h=h)

x_b, y_b = euler_method(func_b, y0=-2, x_range=x_range, h=h)

t_c, x1_c, x2_c = system_c(1, 0, h, x_range)

t_d, x1_d, x2_d = system_d(1, 1, h, x_range)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_a, y_a, label="Approximation (Euler)")
plt.title("1) y' = 0.5 * y, y(0) = 1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_b, y_b, label="Approximation (Euler)", color="orange")
plt.title("2) y' = 2x + 3y, y(0) = -2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_c, x1_c, label="x1 (Euler)")
plt.plot(t_c, x2_c, label="x2 (Euler)")
plt.title("3) System x1' = x2, x2' = -x1")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_d, x1_d, label="x1 (Euler)", color="green")
plt.plot(t_d, x2_d, label="x2 (Euler)", color="red")
plt.title("4) System x1' = x2, x2' = 4x1")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()

plt.tight_layout()
plt.show()
