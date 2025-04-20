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

#Inputs Newton-Raphson
x0 = float(input("Chute Inicial: "))
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

#Newton Raphson
def newton_raphson(f, df, x0, x_tol, y_tol, max_iter):
    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            print("Derivada = 0, sem solucao.")
            return None
        x1 = x0 - fx / dfx
        errox = abs(x1 - x0)
        erro_y = abs(fx)
        print(f"Iteração {i+1}: x = {x1}, errox = {errox:.2e}, erro_y = {erro_y:.2e}")

        if errox < x_tol and erro_y < y_tol:
            print(f"Raiz encontrada: {x1} (apos {i+1} iteracoes)")
            return x1
        x0 = x1
    print("Numero Maximo de Iteracoes Atingida.")
    return None
raiz = newton_raphson(f, df, x0, x_tol, y_tol, max_iter)

# Mostrar a raiz final
if raiz is not None:
    print(f"Raiz final: {raiz:.6f}")
else:
    print("Nenhuma raiz encontrada.")