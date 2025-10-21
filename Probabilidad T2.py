import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def ejercicio_variable_continua():
    print("=== EJERCICIO VARIABLE ALEATORIA CONTINUA ===\n")

    # Función de densidad
    def f(x):
        if 0 < x <= 6:
            return (1 / 72) * x ** 2
        else:
            return 0

    # Función de distribución acumulativa
    def F(x):
        if x <= 0:
            return 0
        elif 0 < x <= 6:
            return (x ** 3) / 216
        else:
            return 1

    # Valor esperado (media)
    def valor_esperado():
        resultado, _ = integrate.quad(lambda x: x * f(x), 0, 6)
        return resultado

    # Varianza
    def varianza():
        E_X = valor_esperado()
        E_X2, _ = integrate.quad(lambda x: (x ** 2) * f(x), 0, 6)
        return E_X2 - E_X ** 2

    # Mostrar resultados
    print("Función de densidad f(x):")
    print("f(x) = (1/72)x², para 0 < x ≤ 6")
    print("f(x) = 0, para otros valores\n")

    print("Función de distribución acumulativa F(x):")
    print("F(x) = 0, para x ≤ 0")
    print("F(x) = x³/216, para 0 < x ≤ 6")
    print("F(x) = 1, para x ≥ 6\n")

    media = valor_esperado()
    var = varianza()

    print(f"Valor medio (μ): {media:.2f}")
    print(f"Varianza (σ²): {var:.4f}")
    print(f"Desviación estándar (σ): {np.sqrt(var):.4f}\n")

    # Calcular P(1 < X < 2)
    prob_1_2 = F(2) - F(1)
    print(f"P(1 < X < 2) = {prob_1_2:.6f}")
    print(f"P(1 < X < 2) = {prob_1_2 * 100:.2f}%\n")

    # Gráficas
    x_vals = np.linspace(-1, 7, 1000)
    f_vals = [f(x) for x in x_vals]
    F_vals = [F(x) for x in x_vals]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, f_vals, 'b-', linewidth=2, label='f(x)')
    plt.fill_between(x_vals, f_vals, alpha=0.3)
    plt.title('Función de Densidad de Probabilidad')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_vals, F_vals, 'r-', linewidth=2, label='F(x)')
    plt.title('Función de Distribución Acumulativa')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Tabla de valores importantes
    print("Tabla de valores:")
    print("x\tf(x)\tF(x)")
    print("-" * 25)
    for x in range(0, 8):
        print(f"{x}\t{f(x):.4f}\t{F(x):.4f}")


# Ejecutar el ejercicio de variable continua
ejercicio_variable_continua()

def ejercicio_dos_dados():
    print("\n" + "=" * 50)
    print("=== EJERCICIO: LANZAMIENTO DE DOS DADOS ===")
    print("=" * 50 + "\n")

    # Calcular todas las combinaciones posibles
    resultados = []
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            suma = d1 + d2
            resultados.append(suma)

    # Función de probabilidad
    def funcion_probabilidad(x):
        if 2 <= x <= 12:
            conteo = resultados.count(x)
            return conteo / len(resultados)
        else:
            return 0

    # Función acumulativa
    def funcion_acumulativa(x):
        if x < 2:
            return 0
        elif x >= 12:
            return 1
        else:
            probabilidad = 0
            for i in range(2, int(x) + 1):
                probabilidad += funcion_probabilidad(i)
            return probabilidad

    # Calcular valor medio
    def valor_medio():
        media = 0
        for x in range(2, 13):
            media += x * funcion_probabilidad(x)
        return media

    # Calcular varianza
    def varianza():
        media = valor_medio()
        var = 0
        for x in range(2, 13):
            var += ((x - media) ** 2) * funcion_probabilidad(x)
        return var

    # Mostrar función de probabilidad
    print("FUNCIÓN DE PROBABILIDAD:")
    print("Suma\tP(X=x)\tFracción")
    print("-" * 30)
    for x in range(2, 13):
        prob = funcion_probabilidad(x)
        fraccion = f"{resultados.count(x)}/36"
        print(f"{x}\t{prob:.4f}\t{fraccion}")

    print("\nFUNCIÓN ACUMULATIVA:")
    print("x\tF(x) = P(X ≤ x)")
    print("-" * 25)
    for x in range(2, 14):
        if x <= 12:
            acum = funcion_acumulativa(x)
            print(f"{x}\t{acum:.4f}")
        else:
            print(f"{x}\t1.0000")

    # Resultados estadísticos
    media = valor_medio()
    var = varianza()
    desviacion = np.sqrt(var)

    print(f"\nRESULTADOS ESTADÍSTICOS:")
    print(f"Valor medio (μ): {media:.2f}")
    print(f"Varianza (σ²): {var:.4f}")
    print(f"Desviación estándar (σ): {desviacion:.4f}")

    # Simulación práctica
    print(f"\nSIMULACIÓN PRÁCTICA (10,000 lanzamientos):")
    np.random.seed(42)  # Para reproducibilidad
    simulaciones = np.random.randint(1, 7, size=(10000, 2))
    sumas_simuladas = np.sum(simulaciones, axis=1)

    media_simulada = np.mean(sumas_simuladas)
    var_simulada = np.var(sumas_simuladas)

    print(f"Media simulada: {media_simulada:.4f}")
    print(f"Varianza simulada: {var_simulada:.4f}")

    # Gráficas
    plt.figure(figsize=(15, 5))

    # Gráfica 1: Función de probabilidad
    plt.subplot(1, 3, 1)
    x_vals = range(2, 13)
    y_prob = [funcion_probabilidad(x) for x in x_vals]
    plt.bar(x_vals, y_prob, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Función de Probabilidad\nSuma de Dos Dados')
    plt.xlabel('Suma de los dados')
    plt.ylabel('P(X = x)')
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Función acumulativa
    plt.subplot(1, 3, 2)
    x_vals_acum = range(2, 14)
    y_acum = [funcion_acumulativa(x) for x in x_vals_acum]
    plt.step(x_vals_acum, y_acum, where='post', color='red', linewidth=2)
    plt.title('Función de Distribución Acumulativa')
    plt.xlabel('x')
    plt.ylabel('F(x) = P(X ≤ x)')
    plt.grid(True, alpha=0.3)

    # Gráfica 3: Histograma de simulación
    plt.subplot(1, 3, 3)
    plt.hist(sumas_simuladas, bins=range(2, 14), density=True,
             alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Histograma de Simulación\n(10,000 lanzamientos)')
    plt.xlabel('Suma de los dados')
    plt.ylabel('Frecuencia relativa')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Probabilidades específicas
    print(f"\nPROBABILIDADES ESPECÍFICAS:")
    print(f"P(X = 7): {funcion_probabilidad(7):.4f} ({funcion_probabilidad(7) * 100:.1f}%)")
    print(f"P(X ≥ 10): {1 - funcion_acumulativa(9):.4f}")
    print(f"P(X ≤ 4): {funcion_acumulativa(4):.4f}")
    print(f"P(5 ≤ X ≤ 9): {funcion_acumulativa(9) - funcion_acumulativa(4):.4f}")


# Ejecutar el ejercicio de dos dados
ejercicio_dos_dados()