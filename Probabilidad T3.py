import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from math import comb

print("=" * 60)
print("APLICACIÓN PYTHON - EJERCICIOS DE PROBABILIDAD CONJUNTA")
print("=" * 60)


# =============================================================================
# EJERCICIO 1: SELECCIÓN DE ESTUDIANTES
# =============================================================================

def ejercicio_estudiantes():
    print("\n" + "=" * 50)
    print("EJERCICIO 1: SELECCIÓN DE ESTUDIANTES")
    print("=" * 50)

    # Parámetros
    total_estudiantes = 8
    sistemas = 3
    electronica = 2
    industrial = 3

    # Calcular función de probabilidad conjunta
    resultados = []
    probabilidades = []

    print("\nFunción de Probabilidad Conjunta:")
    print("-" * 40)
    print(f"{'X':<3} {'Y':<3} {'P(X,Y)':<10} {'Cálculo':<20}")
    print("-" * 40)

    for x in range(3):  # X puede ser 0,1,2
        for y in range(3):  # Y puede ser 0,1,2
            if x + y <= 2:  # Solo 2 estudiantes seleccionados
                industrial_needed = 2 - x - y
                if industrial_needed >= 0 and industrial_needed <= industrial:
                    prob = (comb(sistemas, x) * comb(electronica, y) * comb(industrial, industrial_needed)) / comb(
                        total_estudiantes, 2)
                    resultados.append((x, y))
                    probabilidades.append(prob)
                    print(f"{x:<3} {y:<3} {prob:<10.4f} C(3,{x})·C(2,{y})·C(3,{industrial_needed})/C(8,2)")

    # Calcular P(x+y <= 1)
    prob_r = 0
    print(f"\nP((x,y) ∈ R) donde R = {{(x,y) | x + y ≤ 1}}:")
    print("-" * 50)

    for (x, y), prob in zip(resultados, probabilidades):
        if x + y <= 1:
            prob_r += prob
            print(f"P({x},{y}) = {prob:.4f}")

    print(f"\nP(R) = {prob_r:.4f} = {prob_r * 28:.0f}/28")

    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Distribución conjunta
    x_vals = [r[0] for r in resultados]
    y_vals = [r[1] for r in resultados]

    scatter = ax1.scatter(x_vals, y_vals, s=[p * 2000 for p in probabilidades],
                          c=probabilidades, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('X (Estudiantes de Sistemas)')
    ax1.set_ylabel('Y (Estudiantes de Electrónica)')
    ax1.set_title('Función de Probabilidad Conjunta\n(Tamaño = Probabilidad)')
    ax1.grid(True, alpha=0.3)

    # Añadir etiquetas de probabilidad
    for (x, y), prob in zip(resultados, probabilidades):
        ax1.annotate(f'{prob:.3f}', (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Gráfico 2: Región R
    colors = ['red' if x + y <= 1 else 'blue' for (x, y) in resultados]
    ax2.scatter(x_vals, y_vals, s=[p * 2000 for p in probabilidades],
                c=colors, alpha=0.7)
    ax2.set_xlabel('X (Estudiantes de Sistemas)')
    ax2.set_ylabel('Y (Estudiantes de Electrónica)')
    ax2.set_title('Región R: x + y ≤ 1\n(Rojo = En R, Azul = Fuera de R)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return prob_r


# =============================================================================
# EJERCICIO 2: FUNCIÓN DE DENSIDAD CONJUNTA
# =============================================================================

def ejercicio_densidad():
    print("\n" + "=" * 50)
    print("EJERCICIO 2: FUNCIÓN DE DENSIDAD CONJUNTA")
    print("=" * 50)

    # Definir la función de densidad
    def f(x, y):
        return (2 / 5) * (2 * x + 3 * y)

    # 1. Verificar que es función de densidad
    def integrand(y, x):
        return f(x, y)

    integral, error = integrate.dblquad(integrand, 0, 1, lambda x: 0, lambda x: 1)

    print("1. Verificación de función de densidad:")
    print(f"   ∫∫ f(x,y) dxdy = {integral:.6f}")
    print(f"   Error: {error:.2e}")

    if abs(integral - 1) < 1e-6:
        print("ES una función de densidad válida")
    else:
        print("NO es una función de densidad válida")

    # 2. Calcular P(0 < x ≤ y ≤ 1/2)
    def integrand_R(y, x):
        return f(x, y)

    prob_R, error_R = integrate.dblquad(integrand_R, 0, 1 / 2, lambda y: 0, lambda y: y)

    print(f"\n2. P(0 < x ≤ y ≤ 1/2):")
    print(f"   Resultado: {prob_R:.6f}")
    print(f"   Error: {error_R:.2e}")
    print(f"   En fracción: aproximadamente {prob_R:.6f} = 1/15")

    # Visualización
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Función de densidad
    contour1 = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Función de Densidad f(x,y) = (2/5)(2x + 3y)')
    plt.colorbar(contour1, ax=ax1)

    # Gráfico 2: Región R
    contour2 = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)

    # Dibujar región R: 0 < x ≤ y ≤ 1/2
    y_region = np.linspace(0, 0.5, 100)
    x_region_upper = y_region  # x = y
    x_region_lower = np.zeros_like(y_region)  # x = 0

    ax2.fill_betweenx(y_region, x_region_lower, x_region_upper, alpha=0.5, color='red',
                      label='Región R: 0 < x ≤ y ≤ 1/2')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Región R y Función de Densidad')
    ax2.legend()
    plt.colorbar(contour2, ax=ax2)

    plt.tight_layout()
    plt.show()

    return integral, prob_R


# =============================================================================
# EJERCICIO 3: FUNCIÓN DE PROBABILIDAD DISCRETA
# =============================================================================

def ejercicio_discreto():
    print("\n" + "=" * 50)
    print("EJERCICIO 3: FUNCIÓN DE PROBABILIDAD DISCRETA")
    print("=" * 50)

    # Definir la función de probabilidad
    def f(x, y):
        return (x + y) / 36

    resultados = []
    for x in range(1, 4):
        for y in range(1, 4):
            prob = f(x, y)
            resultados.append((x, y, prob))

    print("Función de Probabilidad Conjunta f(x,y) = (x+y)/36:")
    print("-" * 50)
    print(f"{'x':<3} {'y':<3} {'f(x,y)':<10} {'Cálculo':<15}")
    print("-" * 50)

    for x, y, prob in resultados:
        print(f"{x:<3} {y:<3} {prob:<10.4f} ({x}+{y})/36")

    # a) P(x + y = 4)
    prob_a = sum(prob for x, y, prob in resultados if x + y == 4)

    # b) P(x > y)
    prob_b = sum(prob for x, y, prob in resultados if x > y)

    # c) P(x ≥ 2, y ≤ 1)
    prob_c = sum(prob for x, y, prob in resultados if x >= 2 and y <= 1)

    # d) P(x ≤ 2, y = 1)
    prob_d = sum(prob for x, y, prob in resultados if x <= 2 and y == 1)

    print(f"\na) P(x + y = 4):")
    pares_a = [(x, y) for x, y, prob in resultados if x + y == 4]
    for x, y in pares_a:
        print(f"   f({x},{y}) = {f(x, y):.4f}")
    print(f"   Total = {prob_a:.4f} = {prob_a * 36:.0f}/36")

    print(f"\nb) P(x > y):")
    pares_b = [(x, y) for x, y, prob in resultados if x > y]
    for x, y in pares_b:
        print(f"   f({x},{y}) = {f(x, y):.4f}")
    print(f"   Total = {prob_b:.4f} = {prob_b * 36:.0f}/36")

    print(f"\nc) P(x ≥ 2, y ≤ 1):")
    pares_c = [(x, y) for x, y, prob in resultados if x >= 2 and y <= 1]
    for x, y in pares_c:
        print(f"   f({x},{y}) = {f(x, y):.4f}")
    print(f"   Total = {prob_c:.4f} = {prob_c * 36:.0f}/36")

    print(f"\nd) P(x ≤ 2, y = 1):")
    pares_d = [(x, y) for x, y, prob in resultados if x <= 2 and y == 1]
    for x, y in pares_d:
        print(f"   f({x},{y}) = {f(x, y):.4f}")
    print(f"   Total = {prob_d:.4f} = {prob_d * 36:.0f}/36")

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    casos = [
        ("P(x+y=4)", pares_a, 'red'),
        ("P(x>y)", pares_b, 'blue'),
        ("P(x≥2,y≤1)", pares_c, 'green'),
        ("P(x≤2,y=1)", pares_d, 'orange')
    ]

    for idx, (titulo, pares, color) in enumerate(casos):
        # Todos los puntos
        all_x = [x for x, y, _ in resultados]
        all_y = [y for x, y, _ in resultados]
        all_s = [prob * 1000 for _, _, prob in resultados]

        axes[idx].scatter(all_x, all_y, s=all_s, alpha=0.2, color='gray')

        # Puntos del caso específico
        if pares:
            case_x = [x for x, y in pares]
            case_y = [y for x, y in pares]
            case_probs = [f(x, y) for x, y in pares]
            case_s = [prob * 1000 for prob in case_probs]

            scatter = axes[idx].scatter(case_x, case_y, s=case_s, alpha=0.7, color=color)

            # Etiquetas
            for (x, y), prob in zip(pares, case_probs):
                axes[idx].annotate(f'{prob:.3f}', (x, y), xytext=(5, 5),
                                   textcoords='offset points', fontsize=8)

        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].set_title(titulo)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0.5, 3.5)
        axes[idx].set_ylim(0.5, 3.5)

    plt.tight_layout()
    plt.show()

    return prob_a, prob_b, prob_c, prob_d


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Ejecutar todos los ejercicios
    print("RESOLUCIÓN COMPLETA DE EJERCICIOS")
    print("=" * 60)

    # Ejercicio 1
    resultado1 = ejercicio_estudiantes()

    # Ejercicio 2
    integral, resultado2 = ejercicio_densidad()

    # Ejercicio 3
    resultado3_a, resultado3_b, resultado3_c, resultado3_d = ejercicio_discreto()

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"Ejercicio 1 - P(x+y ≤ 1): {resultado1:.4f} = {resultado1 * 28:.0f}/28")
    print(f"Ejercicio 2 - P(0 < x ≤ y ≤ 1/2): {resultado2:.6f} ≈ 1/15")
    print(f"Ejercicio 3a - P(x+y=4): {resultado3_a:.4f} = {resultado3_a * 36:.0f}/36")
    print(f"Ejercicio 3b - P(x>y): {resultado3_b:.4f} = {resultado3_b * 36:.0f}/36")
    print(f"Ejercicio 3c - P(x≥2,y≤1): {resultado3_c:.4f} = {resultado3_c * 36:.0f}/36")
    print(f"Ejercicio 3d - P(x≤2,y=1): {resultado3_d:.4f} = {resultado3_d * 36:.0f}/36")