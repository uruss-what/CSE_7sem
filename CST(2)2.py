import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def ode_system(x, y): # y[0] = y, y[1] = y'
    dy1 = y[1]
    dy2 = x**2 * y[1] + (2 / x**2) * y[0] + 1 + 4 / x**2
    return np.vstack((dy1, dy2))

def boundary_conditions(ya, yb): # ya когда x=1/2, yb когда x=1
    cond1 = 2 * ya[0] - ya[1] - 6
    cond2 = yb[0] + 3 * yb[1] + 1 
    return np.array([cond1, cond2])

x_values = np.linspace(0.5, 1, 100)  # нач приближение
y_guess = np.zeros((2, x_values.size)) 

solution = solve_bvp(ode_system, boundary_conditions, x_values, y_guess) # численное решение spypy.integrate

if solution.success:
    x_plot = solution.x
    y_plot = solution.y[0]  


    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, label="Решение $y(x)$", color="blue")
    plt.title("Решение краевой задачи")
    plt.xlabel("$x$")
    plt.ylabel("$y(x)$")
    plt.grid()
    plt.legend()
    plt.show()

    result_table = np.column_stack((x_plot, y_plot))
    print("Таблица значений функции y(x):")
    print(" x       y(x)")
    print(result_table)
else:
    print("Решение не удалось найти.")