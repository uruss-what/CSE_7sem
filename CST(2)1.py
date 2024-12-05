import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------Задаем параметры при h^2 > 4km (сильно затухающие колебания) при f=0, f= t -1,f = e^(-е), f = bsin(wt)
m = 1.0  
k = 10.0  
h = 2.0  
gamma = h / (2 * m)  # коэффициент затухания
omega_0 = np.sqrt(k / m)  # собственная частота
omega = np.sqrt(omega_0**2 - gamma**2)  # частота колебаний при затухании


t_span = (0, 10) 
t_eval = np.linspace(*t_span, 500)  # точки времени для численного решения
x0 = 0.0  
v0 = 1.0 


def force_case_0(t):
    return 0

def force_case_1(t):
    return t - 1

def force_case_2(t):
    return np.exp(-t) 

def force_case_3(t, b=1.0, w=2.0):
    return b * np.sin(w * t)


def solve_with_force(force_func, t_span, t_eval, x0, v0):
    def equation_forced(t, y):
        x, v = y
        dxdt = v
        dvdt = -(2 * gamma) * v - omega_0**2 * x + force_func(t) / m
        return [dxdt, dvdt]

    return solve_ivp(equation_forced, t_span, [x0, v0], t_eval=t_eval)

sol_case_0 = solve_with_force(lambda t: 0, t_span, t_eval, x0, v0)
sol_case_1 = solve_with_force(force_case_1, t_span, t_eval, x0, v0)
sol_case_2 = solve_with_force(force_case_2, t_span, t_eval, x0, v0)
sol_case_3 = solve_with_force(lambda t: force_case_3(t, b=1.0, w=omega_0), t_span, t_eval, x0, v0)


plt.figure(figsize=(14, 8))

# f(t) = t - 1
plt.subplot(3, 1, 1)
plt.plot(sol_case_1.t, sol_case_1.y[0], label="Численное решение", color='blue')
plt.plot(sol_case_0.t, sol_case_0.y[0], label="Решение (f(t) = 0)", linestyle='--', color='black')
plt.title("Движение груза при $f(t) = t - 1$", fontsize=14)
plt.xlabel("Время, сек", fontsize=12)
plt.ylabel("Смещение, м", fontsize=12)
plt.grid()
plt.legend()

# f(t) = e^(-t)
plt.subplot(3, 1, 2)
plt.plot(sol_case_2.t, sol_case_2.y[0], label="Численное решение", color='green')
plt.plot(sol_case_0.t, sol_case_0.y[0], label="Решение (f(t) = 0)", linestyle='--', color='black')
plt.title("Движение груза при $f(t) = e^{-t}$", fontsize=14)
plt.xlabel("Время, сек", fontsize=12)
plt.ylabel("Смещение, м", fontsize=12)
plt.grid()
plt.legend()

# f(t) = b*sin(wt)
plt.subplot(3, 1, 3)
plt.plot(sol_case_3.t, sol_case_3.y[0], label="Численное решение", color='red')
plt.plot(sol_case_0.t, sol_case_0.y[0], label="Решение (f(t) = 0)", linestyle='--', color='black')
plt.title("Движение груза при $f(t) = b \sin(\omega t)$", fontsize=14)
plt.xlabel("Время, сек", fontsize=12)
plt.ylabel("Смещение, м", fontsize=12)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------Задаем новые параметры для h^2 > 4km(слабо затухающие колебания) при f=0, f= t -1,f = e^(-е), f = bsin(wt)
h_new = 10.0  # увеличиваем коэффициент сопротивления
gamma_new = h_new / (2 * m)  # новый коэффициент затухания
omega_0_new = omega_0 


def solve_with_force_overdamped(force_func, t_span, t_eval, x0, v0):
    def equation_overdamped(t, y):
        x, v = y
        dxdt = v
        dvdt = -(2 * gamma_new) * v - omega_0_new**2 * x + force_func(t) / m
        return [dxdt, dvdt]

    return solve_ivp(equation_overdamped, t_span, [x0, v0], t_eval=t_eval)


sol_case_0_new = solve_with_force_overdamped(lambda t: 0, t_span, t_eval, x0, v0) 
sol_case_1_new = solve_with_force_overdamped(force_case_1, t_span, t_eval, x0, v0)
sol_case_2_new = solve_with_force_overdamped(force_case_2, t_span, t_eval, x0, v0)
sol_case_3_new = solve_with_force_overdamped(lambda t: force_case_3(t, b=1.0, w=omega_0), t_span, t_eval, x0, v0)


plt.figure(figsize=(14, 8))

# f(t) = t - 1
plt.subplot(3, 1, 1)
plt.plot(sol_case_1_new.t, sol_case_1_new.y[0], label="Численное решение (f(t) = t - 1)", color='blue')
plt.plot(sol_case_0_new.t, sol_case_0_new.y[0], label="Решение (f(t) = 0)", linestyle='--', color='black')
plt.title("Движение груза при $f(t) = t - 1$ и $h^2 > 4km$", fontsize=14)
plt.xlabel("Время, сек", fontsize=12)
plt.ylabel("Смещение, м", fontsize=12)
plt.grid()
plt.legend()

# f(t) = e^(-t)
plt.subplot(3, 1, 2)
plt.plot(sol_case_2_new.t, sol_case_2_new.y[0], label="Численное решение (f(t) = e^{-t})", color='green')
plt.plot(sol_case_0_new.t, sol_case_0_new.y[0], label="Решение (f(t) = 0)", linestyle='--', color='black')
plt.title("Движение груза при $f(t) = e^{-t}$ и $h^2 > 4km$", fontsize=14)
plt.xlabel("Время, сек", fontsize=12)
plt.ylabel("Смещение, м", fontsize=12)
plt.grid()
plt.legend()

# f(t) = b*sin(wt)
plt.subplot(3, 1, 3)
plt.plot(sol_case_3_new.t, sol_case_3_new.y[0], label="Численное решение (f(t) = b\\sin(\\omega t))", color='red')
plt.plot(sol_case_0_new.t, sol_case_0_new.y[0], label="Решение (f(t) = 0)", linestyle='--', color='black')
plt.title("Движение груза при $f(t) = b \\sin(\\omega t)$ и $h^2 > 4km$", fontsize=14)
plt.xlabel("Время, сек", fontsize=12)
plt.ylabel("Смещение, м", fontsize=12)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
