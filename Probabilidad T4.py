import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def ejercicio_bombillas():
    """Solución del ejercicio de tiempo de vida de bombillas"""
    # Parámetros
    mu = 800  # media
    sigma = 40  # desviación estándar
    lim_inf = 750
    lim_sup = 850

    # Crear distribución normal
    distribucion = stats.norm(mu, sigma)

    # Calcular probabilidad
    prob = distribucion.cdf(lim_sup) - distribucion.cdf(lim_inf)
    prob_porcentaje = prob * 100

    # Calcular valores Z
    z1 = (lim_inf - mu) / sigma
    z2 = (lim_sup - mu) / sigma

    # Mostrar resultados
    print("=== EJERCICIO 1: TIEMPO DE VIDA DE BOMBILLAS ===")
    print(f"Media: {mu} horas")
    print(f"Desviación estándar: {sigma} horas")
    print(f"Límite inferior: {lim_inf} horas")
    print(f"Límite superior: {lim_sup} horas")
    print(f"Valor Z inferior: {z1:.2f}")
    print(f"Valor Z superior: {z2:.2f}")
    print(f"Probabilidad: {prob:.4f} ({prob_porcentaje:.2f}%)")

    # Graficar
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    y = distribucion.pdf(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Distribución Normal')
    plt.fill_between(x, y, where=(x >= lim_inf) & (x <= lim_sup),
                     color='red', alpha=0.3, label=f'Zona de interés ({prob_porcentaje:.1f}%)')
    plt.axvline(mu, color='green', linestyle='--', label=f'Media ({mu})')
    plt.axvline(lim_inf, color='orange', linestyle='--', label=f'Límite inferior ({lim_inf})')
    plt.axvline(lim_sup, color='orange', linestyle='--', label=f'Límite superior ({lim_sup})')
    plt.title('Distribución del Tiempo de Vida de Bombillas')
    plt.xlabel('Horas de duración')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return prob




def ejercicio_puntajes_examen():
    """Solución del ejercicio de puntajes de examen"""
    # Parámetros
    mu = 500
    sigma = 100
    percentil = 0.85  # 15% superior = percentil 85

    # Calcular valor Z para el percentil 85
    z = stats.norm.ppf(percentil)

    # Calcular puntaje mínimo
    puntaje_minimo = mu + z * sigma

    # Crear distribución
    distribucion = stats.norm(mu, sigma)

    print("\n=== EJERCICIO 2: PUNTAJES DE EXAMEN ===")
    print(f"Media: {mu} puntos")
    print(f"Desviación estándar: {sigma} puntos")
    print(f"Percentil requerido: {percentil * 100}%")
    print(f"Valor Z correspondiente: {z:.3f}")
    print(f"Puntaje mínimo requerido: {puntaje_minimo:.1f} puntos")

    # Graficar
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    y = distribucion.pdf(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Distribución de Puntajes')

    # Área de aceptación (15% superior)
    x_aceptacion = np.linspace(puntaje_minimo, mu + 3 * sigma, 500)
    y_aceptacion = distribucion.pdf(x_aceptacion)
    plt.fill_between(x_aceptacion, y_aceptacion, color='green', alpha=0.3,
                     label=f'Zona de aceptación (15% superior)')

    plt.axvline(puntaje_minimo, color='red', linestyle='--',
                label=f'Puntaje mínimo ({puntaje_minimo:.1f})')
    plt.axvline(mu, color='orange', linestyle='--', label=f'Media ({mu})')

    plt.title('Distribución de Puntajes - Criterio de Aceptación')
    plt.xlabel('Puntaje')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return puntaje_minimo



def ejercicio_llegadas_banco():
    """Solución del ejercicio de llegadas al banco"""
    # Parámetros
    lambda_val = 5  # promedio de llegadas por hora
    k = 3  # número exacto de llegadas que nos interesa

    # Crear distribución Poisson
    distribucion = stats.poisson(lambda_val)

    # Calcular probabilidad
    prob_exacta = distribucion.pmf(k)
    prob_acumulada = distribucion.cdf(k)

    print("\n=== EJERCICIO 3: LLEGADAS AL BANCO ===")
    print(f"Promedio de llegadas por hora: {lambda_val}")
    print(f"Probabilidad de exactamente {k} llegadas: {prob_exacta:.4f} ({prob_exacta * 100:.2f}%)")
    print(f"Probabilidad acumulada hasta {k} llegadas: {prob_acumulada:.4f} ({prob_acumulada * 100:.2f}%)")

    # Graficar distribución Poisson
    valores_k = np.arange(0, 15)  # Valores posibles de k
    probabilidades = distribucion.pmf(valores_k)

    plt.figure(figsize=(12, 6))

    # Gráfico de barras
    plt.subplot(1, 2, 1)
    bars = plt.bar(valores_k, probabilidades, color='skyblue', alpha=0.7, edgecolor='black')
    plt.bar(k, distribucion.pmf(k), color='red', alpha=0.8, label=f'P(X={k})')
    plt.xlabel('Número de llegadas (k)')
    plt.ylabel('Probabilidad P(X=k)')
    plt.title(f'Distribución Poisson (λ={lambda_val})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico de probabilidad acumulada
    plt.subplot(1, 2, 2)
    prob_acum = distribucion.cdf(valores_k)
    plt.step(valores_k, prob_acum, where='post', color='green', linewidth=2)
    plt.axvline(k, color='red', linestyle='--', alpha=0.7)
    plt.axhline(prob_acumulada, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Número de llegadas (k)')
    plt.ylabel('Probabilidad acumulada P(X≤k)')
    plt.title('Función de Distribución Acumulada')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Mostrar tabla de probabilidades
    print(f"\nTabla de probabilidades para λ={lambda_val}:")
    print("k\tP(X=k)\t\tP(X≤k)")
    print("-" * 35)
    for i in range(10):
        prob_i = distribucion.pmf(i)
        prob_acum_i = distribucion.cdf(i)
        print(f"{i}\t{prob_i:.4f}\t\t{prob_acum_i:.4f}")

    return prob_exacta




def ejercicio_defectos_tejidos():
    """Solución del ejercicio de defectos en tejidos"""
    # Parámetros originales
    defectos_10m = 2
    longitud_estudio = 5  # metros

    # Calcular lambda para 5 metros
    lambda_val = defectos_10m * (longitud_estudio / 10)

    # Crear distribución Poisson
    distribucion = stats.poisson(lambda_val)

    # Calcular probabilidades
    prob_ninguno = distribucion.pmf(0)
    prob_al_menos_uno = 1 - prob_ninguno

    print("\n=== EJERCICIO 4: DEFECTOS EN TEJIDOS ===")
    print(f"Defectos en 10m: {defectos_10m}")
    print(f"Longitud a estudiar: {longitud_estudio}m")
    print(f"Lambda para {longitud_estudio}m: {lambda_val}")
    print(f"Probabilidad de 0 defectos: {prob_ninguno:.6f} ({prob_ninguno * 100:.2f}%)")
    print(f"Probabilidad de al menos 1 defecto: {prob_al_menos_uno:.6f} ({prob_al_menos_uno * 100:.2f}%)")

    # Graficar
    valores_k = np.arange(0, 8)
    probabilidades = distribucion.pmf(valores_k)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(valores_k, probabilidades, color='lightcoral', alpha=0.7, edgecolor='black')

    # Resaltar la barra de 0 defectos
    plt.bar(0, distribucion.pmf(0), color='red', alpha=0.8,
            label=f'P(X=0) = {prob_ninguno:.4f}')

    # Marcar el área de "al menos 1 defecto"
    for i in range(1, len(valores_k)):
        plt.bar(i, distribucion.pmf(i), color='green', alpha=0.6)

    plt.xlabel('Número de defectos')
    plt.ylabel('Probabilidad P(X=k)')
    plt.title(f'Distribución de Defectos en {longitud_estudio}m de Tela (λ={lambda_val})')
    plt.legend(['Probabilidad de 0 defectos', 'Probabilidad de k defectos (k≥1)'])
    plt.grid(True, alpha=0.3)

    # Añadir anotaciones
    plt.annotate(f'P(X=0) = {prob_ninguno:.4f}',
                 xy=(0, distribucion.pmf(0)),
                 xytext=(2, distribucion.pmf(0) + 0.02),
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.annotate(f'P(X≥1) = {prob_al_menos_uno:.4f}',
                 xy=(2, distribucion.pmf(2)),
                 xytext=(3, distribucion.pmf(2) + 0.02),
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.show()

    return prob_al_menos_uno




if __name__ == "__main__":
    print("APLICACIONES DE DISTRIBUCIÓN NORMAL Y POISSON EN PYTHON")
    print("=" * 60)

    # Ejecutar todos los ejercicios
    resultados = {}

    resultados['bombillas'] = ejercicio_bombillas()
    resultados['puntajes'] = ejercicio_puntajes_examen()
    resultados['llegadas'] = ejercicio_llegadas_banco()
    resultados['defectos'] = ejercicio_defectos_tejidos()

    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS:")
    print(f"• Probabilidad bombillas entre 750-850h: {resultados['bombillas']:.2%}")
    print(f"• Puntaje mínimo para examen: {resultados['puntajes']:.1f} puntos")
    print(f"• Probabilidad exactamente 3 llegadas: {resultados['llegadas']:.2%}")
    print(f"• Probabilidad al menos 1 defecto: {resultados['defectos']:.2%}")