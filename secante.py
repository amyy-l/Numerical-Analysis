import numpy as np
import matplotlib.pyplot as plt  
import sympy as sp


# Definicao de simbolos 
x = sp.symbols ('x')
y = sp.symbols ('y')
z = sp.symbols ('z')

#Input de funcao
user_input = input("Digite a funcao em X; (Ex.: x**2 -2):  ")
f_expr = sp.sympify(user_input)

#Primeira e Segunda Derivada
df_expr = sp.diff(f_expr, x)
ddf_expr = sp.diff(df_expr, x)

#Converter para funcoes numericas
f = sp.lambdify(x, f_expr, 'numpy')
df = sp.lambdify(x, df_expr, 'numpy')
ddf = sp.lambdify(x, ddf_expr, 'numpy')

#Definicao intervalo e plot do grafico
a = float(input("Entre o valor inicial do range (a): "))
b = float(input("Enter o valor final do range (b): "))
xvalues = np.linspace(a, b, 100)
yvalues = f(xvalues)
plt.plot(xvalues, yvalues, label='f(x)')
plt.axhline(0, color='blue', linestyle='--')
plt.title('Grafico f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

#Inputs Secante
x0 = float(input("Chute Inicial x0: "))
x1 = float(input("Chute Inicial x1: "))
x_tol = float(input("Defina tolerancia em x (ex.: 1e-6): "))
y_tol = float(input("Defina tolerancia em y (ex.: 1e-6): "))
max_iter = int(input("Defina o numero maximo de iteracoes (ex.: 100): "))

#Teste da Convergencia
try:
    yvalues = abs(f(x0) * ddf(x0) / (df(x0) ** 2))
    print(f"Condicao de convergencia <=1: {yvalues:.4f}")
    if yvalues >= 1:
        print("⚠️ Nao ha garantia de convergencia neste intervalo.")
except ZeroDivisionError:
    print("⚠️ Derivada = 0 no chute inicial. Nao e possivel seguir.")
    exit()

#Secante
def secant_method(f, x0, x1, x_tol, y_tol, max_iter):
    for i in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)

        if f_x1 - f_x0 == 0:
            print("Divisão por zero na fórmula da secante.")
            return None

        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        errox = abs(x2 - x1)
        erro_y = abs(f(x2))
        print(f"Iteração {i+1}: x = {x2}, errox = {errox:.2e}, erro_y = {erro_y:.2e}")

        if errox < x_tol and erro_y < y_tol:
            print(f"Raiz encontrada: {x2} (após {i+1} iterações)")
            return x2

        x0, x1 = x1, x2

    print("Máximo de iterações atingido.")
    return None

raiz = secant_method(f, x0, x1, x_tol, y_tol, max_iter)

# Mostrar a raiz final
if raiz is not None:
    print(f"Raiz final: {raiz:.6f}")
else:
    print("Nenhuma raiz encontrada.")

