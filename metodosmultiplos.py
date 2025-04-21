import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')

def get_function():
    user_input = input("Digite a função f(x) = 0: ")
    f_expr = sp.sympify(user_input)
    f = sp.lambdify(x, f_expr, 'numpy')
    return f_expr, f

def plot_function(f, a, b, raiz=None):
    x_vals = np.linspace(a, b, 400)
    y_vals = f(x_vals)
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(x_vals, y_vals, label='f(x)')
    if raiz is not None:
        plt.plot(raiz, f(raiz), 'ro', label=f'Raiz ≈ {raiz:.4f}')
    plt.title('Gráfico da Função')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

def bisseccao(f, a, b, x_tol, y_tol, max_iter):
    if f(a) * f(b) > 0:
        print("⚠️ A função não muda de sinal no intervalo. Bissecção não é aplicável.")
        return None
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

def newton_raphson(f, df, ddf, x0, x_tol, y_tol, max_iter):
    try:
        conv = abs(f(x0) * ddf(x0) / (df(x0) ** 2))
        print(f"Cond. convergência <= 1: {conv:.4f}")
        if conv >= 1:
            print("⚠️ Pode não convergir.")
    except:
        print("⚠️ Derivada zero no chute inicial.")
        return None
    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            print("Derivada = 0. Sem solução.")
            return None
        x1 = x0 - fx / dfx
        errox = abs(x1 - x0)
        erro_y = abs(fx)
        print(f"Iteração {i+1}: x = {x1}, errox = {errox:.2e}, erro_y = {erro_y:.2e}")
        if errox < x_tol and erro_y < y_tol:
            print(f"Raiz encontrada: {x1} (após {i+1} iterações)")
            return x1
        x0 = x1
    print("Máximo de iterações atingido.")
    return None

def secante(f, x0, x1, x_tol, y_tol, max_iter):
    for i in range(max_iter):
        f_x0, f_x1 = f(x0), f(x1)
        if f_x1 - f_x0 == 0:
            print("Divisão por zero.")
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

def ponto_fixo(f_expr, f, a, b, x0, x_tol, y_tol, max_iter):
    df_expr = sp.diff(f_expr, x)
    df = sp.lambdify(x, df_expr, 'numpy')
    df_vals = np.abs(df(np.linspace(a, b, 400)))
    lambda_val = 1 / np.max(df_vals)
    g_expr = x - lambda_val * f_expr
    g = sp.lambdify(x, g_expr, 'numpy')
    dg_expr = sp.diff(g_expr, x)
    dg = sp.lambdify(x, dg_expr, 'numpy')
    print(f"λ = {lambda_val:.4f}, |g'(x0)| = {abs(dg(x0)):.4f}")
    if abs(dg(x0)) >= 1:
        print("⚠️ Método pode não convergir.")
    for i in range(max_iter):
        x1 = g(x0)
        errox = abs(x1 - x0)
        erro_y = abs(f(x1))
        print(f"Iteração {i+1}: x = {x1}, errox = {errox:.2e}, erro_y = {erro_y:.2e}")
        if errox < x_tol and erro_y < y_tol:
            print(f"Raiz encontrada: {x1} (após {i+1} iterações)")
            return x1
        x0 = x1
    print("Máximo de iterações atingido.")
    return None

# Menu interativo
print("\n===== MÉTODOS NUMÉRICOS =====")
print("1 - Bissecção")
print("2 - Newton-Raphson")
print("3 - Secante")
print("4 - Ponto Fixo")
metodo = input("Escolha o método (1-4): ")

f_expr, f = get_function()
a = float(input("Intervalo inicial (a): "))
b = float(input("Intervalo final (b): "))
plot_function(f, a, b)

x_tol = float(input("Tolerância em x: "))
y_tol = float(input("Tolerância em y: "))
max_iter = int(input("Máximo de iterações: "))

raiz = None

if metodo == '1':
    raiz = bisseccao(f, a, b, x_tol, y_tol, max_iter)

elif metodo == '2':
    x0 = float(input("Chute inicial x0: "))
    df_expr = sp.diff(f_expr, x)
    ddf_expr = sp.diff(df_expr, x)
    df = sp.lambdify(x, df_expr, 'numpy')
    ddf = sp.lambdify(x, ddf_expr, 'numpy')
    raiz = newton_raphson(f, df, ddf, x0, x_tol, y_tol, max_iter)

elif metodo == '3':
    x0 = float(input("Chute inicial x0: "))
    x1 = float(input("Chute inicial x1: "))
    raiz = secante(f, x0, x1, x_tol, y_tol, max_iter)

elif metodo == '4':
    x0 = float(input("Chute inicial x0: "))
    raiz = ponto_fixo(f_expr, f, a, b, x0, x_tol, y_tol, max_iter)

else:
    print("Opção inválida.")

if raiz is not None:
    plot_function(f, a, b, raiz)
    print(f"Raiz final: {raiz:.6f}")
else:
    print("Nenhuma raiz encontrada.")
