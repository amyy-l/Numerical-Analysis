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
#Converter para funcoes numericas
f = sp.lambdify(x, f_expr, 'numpy')

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

#Inputs Bisseccao
x_tol = float(input("Defina tolerancia em x (ex.: 1e-6): "))
y_tol = float(input("Defina tolerancia em y (ex.: 1e-6): "))
max_iter = int(input("Defina o numero maximo de iteracoes (ex.: 100): "))

#Metodo Bisseccao
def bisseccao(f, a, b, x_tol, y_tol, max_iter):
    for i in range(max_iter):
        m = (a + b) / 2
        fm = f(m)
        errox = abs(b - a) / 2
        erro_y = abs(fm)

        print(f"Iteração {i+1}: x = {m}, errox = {errox:.2e}, erro_y = {erro_y:.2e}")

        if errox < x_tol or erro_y < y_tol:
            print(f"Raiz encontrada: {m} (após {i+1} iterações)")
            return m

        if f(a) * fm < 0:
            b = m
        else:
            a = m

    print("Máximo de iterações atingido.")
    return None

# Run the method
raiz = bisseccao(f, a, b, x_tol, y_tol, max_iter)

# Show final result
if raiz is not None:
    print(f"Raiz final: {raiz:.6f}")
else:
    print("Nenhuma raiz encontrada.")