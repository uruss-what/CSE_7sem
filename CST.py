import math
import numpy as np
import matplotlib.pyplot as plt

intervals = [(0, np.pi /2), (2, 10), (-3, 3)]
steps = [0.01, 0.005, 0.001]

def func1(x):
    return np.exp(-x**2)

def func2(x):
    return np.sin(3 * x)

def func3(x):
    return np.cos(5 * x)**2

def func4(x, terms=10):
    return sum([(x ** (2 * n)) / math.factorial(2 * n) for n in range(terms)])

functions = [func1, func2, func3, func4]
function_names = ["e^(-x^2)", "sin(3x)", "cos^2(5x)", "sum(x^(2n)/(2n)!)"]

for i,func in enumerate(functions):
  for a, b in intervals:
    for h in steps:
        x = np.arange(a, b + h, h)
        y = np.vectorize(func)(x)
        
        x_orig = np.linspace(a,b,1000)
        y_orig = np.vectorize(func)(x_orig)
        

        plt.figure(figsize=(12, 6))
        plt.plot(x_orig,y_orig,label="Original function",color="black")
        plt.plot(x, y,'o', label=f"f, h={h}", markersize=2, color="red")
        
        
        plt.title(f"f {function_names[i]} [{a}, {b}]") 
        plt.xlabel("x")
        plt.ylabel("x")
        plt.legend()

        plt.show()

