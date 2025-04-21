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
f = sp.lambdify(x, f_expr, 'numpy')

# Derivada de f(x)
df_expr = sp.diff(f_expr, x)
df = sp.lambdify(x, df_expr, 'numpy')

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


# Chute inicial e tolerâncias
x0 = float(input("Chute Inicial x0: "))
x_tol = float(input("Tolerância em x (ex: 1e-6): "))
y_tol = float(input("Tolerância em y (ex: 1e-6): "))
max_iter = int(input("Número máximo de iterações: "))

# Calcula valor máximo de |f'(x)| no intervalo [a, b]
x_vals = np.linspace(a, b, 400)
df_vals = np.abs(df(x_vals))
df_max = np.max(df_vals)

# Definir faotr relaxamento (lambda)
lambda_val = 1 / df_max
print(f"λ automático calculado: {lambda_val:.4f}")

# Define g(x) = x - lambda * f(x)
g_expr = x - lambda_val * f_expr
g = sp.lambdify(x, g_expr, 'numpy')

# Derivada de g(x)
dg_expr = sp.diff(g_expr, x)
dg = sp.lambdify(x, dg_expr, 'numpy')


# Teste de convergência
dg_val = abs(dg(x0))
print(f"|g'(x0)| = {dg_val:.4f}")
if dg_val >= 1:
    print("⚠️ Aviso: |g'(x0)| >= 1. O método pode não convergir.")

# Método do ponto fixo
def fixed_point(g, f, x0, x_tol, y_tol, max_iter):
    for i in range(max_iter):
        x1 = g(x0)
        errox = abs(x1 - x0)
        erroy = abs(f(x1))

        print(f"Iteração {i+1}: x = {x1}, errox = {errox:.2e}, erroy = {erroy:.2e}")

        if errox < x_tol and erroy < y_tol:
            print(f"Raiz encontrada: {x1} (após {i+1} iterações)")
            return x1

        x0 = x1

    print("Máximo de iterações atingido.")
    return None


# Rodar método
raiz = fixed_point(g, f, x0, x_tol, y_tol, max_iter)

# Mostrar resultado
if raiz is not None:
    print(f"Raiz final: {raiz:.6f}")
else:
    print("Nenhuma raiz encontrada.")