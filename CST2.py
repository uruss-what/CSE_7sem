import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.exp((-x**2)/2)

def f2(x):
    return np.sin((3 * (x**4)) / 5)**3

def f3(x):
    return np.cos((x / (x + 1)))**2

def f4(x):
    return np.log(x + np.sqrt(4 + x**2))

def f5(x):
    return x * np.arctan(2 * x) / (x**2 + 4)



def df1(x):
    return -x * np.exp((-x**2) /2)

def df2(x):
    return 12 * x**3 * np.sin((3 * x**4) / 5)**2 * np.cos((3 * x**4) / 5) / 5

def df3(x):
    return -2 * np.cos(x / (x + 1)) * np.sin(x / (x + 1)) * (1 / (x + 1)**2)

def df4(x):
    return (1 + x / np.sqrt(4 + x**2)) / (x + np.sqrt(4 + x**2))

def df5(x):
    numerator = np.arctan(2 * x) + (2 * x) / (1 + (2 * x) ** 2)
    denominator = x ** 2 + 4
    return (numerator * denominator - x * np.arctan(2 * x) * 2 * x) / (denominator ** 2)


functions = [f1, f2, f3, f4, f5]
dfunctions = [df1, df2, df3, df4, df5]
function_names = ['e^(-x/2)','sin^3((3x^4)/5)','cos^2(x/(x+1))','ln(x+ sqrt(4 + x^2))','x * arctan(2 * x) / (x^2 + 4)']


intervals = [(0, 1), (2, 15), (-5, 5)]
steps = [0.01, 0.005]


def num_derivative(f, x, h):
    return (f(x + h) - f(x)) / h


for i, (func, num_derivative) in enumerate(zip(functions, dfunctions)):
    for a,b in intervals:
        for h in steps:
            x = np.arange(a, b + h, h)
            y = func(x)
            dy_num = (y[1:] - y[:-1]) / h
            x_num = x[:-1]
            
            dy_anal = num_derivative(x)

            
            plt.figure(figsize=(12, 6))
            plt.plot(x_num, dy_num, label="Численная производная", color="black")
            plt.plot(x, dy_anal, label="Аналитическая производная", color="red")
            plt.title(f"Графики производных для функции {function_names[i]} на отрезке [{a}, {b}], шаг {h}")
            plt.xlabel("x")
            plt.ylabel("dy/dx")
            plt.legend()
            plt.grid()
            plt.show()
