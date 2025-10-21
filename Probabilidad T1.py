from math import comb, factorial, exp
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("SOLUCIÓN DE EJERCICIOS DE PROBABILIDAD")
print("=" * 60)

# =============================================================================
# EJERCICIO 1: Selección de estudiantes
# =============================================================================
print("\n" + "="*50)
print("EJERCICIO 1: Selección de estudiantes")
print("="*50)

# Datos del problema
total_estudiantes = 20
electronica = 8
sistemas = 3
industrial = 9

# a) Sin sustitución
print("\n--- SIN SUSTITUCIÓN ---")
espacio_muestral_sin = comb(total_estudiantes, 3)

# 1. Los 3 estudiantes sean de Electrónica
prob_1a = comb(electronica, 3) / espacio_muestral_sin
print(f"1. P(3 Electrónica) = {prob_1a:.4f} ({comb(electronica, 3)}/{espacio_muestral_sin})")

# 2. Los 3 estudiantes sean de Sistemas
prob_2a = comb(sistemas, 3) / espacio_muestral_sin
print(f"2. P(3 Sistemas) = {prob_2a:.6f} ({comb(sistemas, 3)}/{espacio_muestral_sin})")

# 3. 2 de Electrónica y 1 de Sistemas
prob_3a = (comb(electronica, 2) * comb(sistemas, 1)) / espacio_muestral_sin
print(f"3. P(2 Electrónica, 1 Sistemas) = {prob_3a:.4f} (84/{espacio_muestral_sin})")

# 4. Al menos 1 sea de Sistemas
prob_ningun_sistema = comb(total_estudiantes - sistemas, 3) / espacio_muestral_sin
prob_4a = 1 - prob_ningun_sistema
print(f"4. P(al menos 1 Sistemas) = {prob_4a:.4f}")

# 5. Se escoja 1 de cada carrera
prob_5a = (comb(electronica, 1) * comb(sistemas, 1) * comb(industrial, 1)) / espacio_muestral_sin
print(f"5. P(1 de cada carrera) = {prob_5a:.4f} (216/{espacio_muestral_sin})")

# 6. En orden: Electrónica-Sistemas-Industrial
espacio_ordenado = total_estudiantes * (total_estudiantes - 1) * (total_estudiantes - 2)
prob_6a = (electronica * sistemas * industrial) / espacio_ordenado
print(f"6. P(Orden E-S-I) = {prob_6a:.4f} (216/{espacio_ordenado})")

# b) Con sustitución
print("\n--- CON SUSTITUCIÓN ---")
espacio_muestral_con = total_estudiantes ** 3

# 1. Los 3 estudiantes sean de Electrónica
prob_1b = (electronica/total_estudiantes) ** 3
print(f"1. P(3 Electrónica) = {prob_1b:.4f}")

# 2. Los 3 estudiantes sean de Sistemas
prob_2b = (sistemas/total_estudiantes) ** 3
print(f"2. P(3 Sistemas) = {prob_2b:.6f}")

# 3. 2 de Electrónica y 1 de Sistemas
prob_3b = 3 * (electronica/total_estudiantes)**2 * (sistemas/total_estudiantes)
print(f"3. P(2 Electrónica, 1 Sistemas) = {prob_3b:.4f}")

# 4. Al menos 1 sea de Sistemas
prob_4b = 1 - ((total_estudiantes - sistemas)/total_estudiantes) ** 3
print(f"4. P(al menos 1 Sistemas) = {prob_4b:.6f}")

# 5. Se escoja 1 de cada carrera
prob_5b = 6 * (electronica/total_estudiantes) * (sistemas/total_estudiantes) * (industrial/total_estudiantes)
print(f"5. P(1 de cada carrera) = {prob_5b:.4f}")

# 6. En orden: Electrónica-Sistemas-Industrial
prob_6b = (electronica/total_estudiantes) * (sistemas/total_estudiantes) * (industrial/total_estudiantes)
print(f"6. P(Orden E-S-I) = {prob_6b:.4f}")

# =============================================================================
# EJERCICIO 2: Arreglo de libros
# =============================================================================
print("\n" + "="*50)
print("EJERCICIO 2: Arreglo de libros")
print("="*50)

ingenieria = 4
ingles = 6
fisica = 2
total_libros = ingenieria + ingles + fisica

# a) Libros de cada asignatura juntos
permutaciones_bloques = factorial(3)
permutaciones_internas = factorial(ingenieria) * factorial(ingles) * factorial(fisica)
total_a = permutaciones_bloques * permutaciones_internas
print(f"a) Libros de cada asignatura juntos: {total_a:,} formas")

# b) Solo libros de Ingeniería juntos
# Tratamos los 4 libros de ingeniería como un bloque
bloque_ingenieria = 1
otros_libros = ingles + fisica
total_objetos = bloque_ingenieria + otros_libros
permutaciones_externas = factorial(total_objetos)
permutaciones_internas_ing = factorial(ingenieria)
total_b = permutaciones_externas * permutaciones_internas_ing
print(f"b) Solo libros de Ingeniería juntos: {total_b:,} formas")

# =============================================================================
# EJERCICIO 3: Formación de comité
# =============================================================================
print("\n" + "="*50)
print("EJERCICIO 3: Formación de comité")
print("="*50)

ingenieros_total = 5
abogados_total = 7

# a) Cualquier ingeniero y cualquier abogado
comite_a = comb(ingenieros_total, 2) * comb(abogados_total, 3)
print(f"a) Cualquier ingeniero y abogado: {comite_a} formas")

# b) Abogado determinado debe pertenecer
comite_b = comb(ingenieros_total, 2) * comb(abogados_total - 1, 2)
print(f"b) Abogado determinado incluido: {comite_b} formas")

# c) Ingenieros determinados no pueden pertenecer (asumiendo 2 excluidos)
ingenieros_disponibles = ingenieros_total - 2
comite_c = comb(ingenieros_disponibles, 2) * comb(abogados_total, 3)
print(f"c) 2 ingenieros excluidos: {comite_c} formas")

# =============================================================================
# EJERCICIO 4: Ordenamiento de estudiantes
# =============================================================================
print("\n" + "="*50)
print("EJERCICIO 4: Ordenamiento de estudiantes")
print("="*50)

electronica_4 = 5
sistemas_4 = 2
industrial_4 = 3
total_estudiantes_4 = electronica_4 + sistemas_4 + industrial_4

# Permutaciones con repetición
permutaciones = factorial(total_estudiantes_4)
repeticiones = factorial(electronica_4) * factorial(sistemas_4) * factorial(industrial_4)
formas_ordenar = permutaciones / repeticiones
print(f"Formas de ordenar estudiantes: {formas_ordenar:,}")

# =============================================================================
# EJERCICIO 5: Probabilidades con dados
# =============================================================================
print("\n" + "="*50)
print("EJERCICIO 5: Probabilidades con dados")
print("="*50)

# a) No obtener 7 u 11 en dos lanzamientos
# Probabilidad de obtener 7 u 11 en un lanzamiento
# Combinaciones que suman 7: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1) -> 6
# Combinaciones que suman 11: (5,6),(6,5) -> 2
# Total combinaciones: 36
prob_7_u_11 = (6 + 2) / 36
prob_no_7_u_11 = 1 - prob_7_u_11
prob_5a = prob_no_7_u_11 ** 2
print(f"a) P(no 7 u 11 en 2 lanzamientos) = {prob_5a:.4f}")

# b) Obtener tres 6 en 5 lanzamientos
n_lanzamientos = 5
k_exitos = 3
p_exito = 1/6
prob_5b = comb(n_lanzamientos, k_exitos) * (p_exito ** k_exitos) * ((1 - p_exito) ** (n_lanzamientos - k_exitos))
print(f"b) P(3 seis en 5 lanzamientos) = {prob_5b:.6f}")

# =============================================================================
# EJERCICIO 6: Memorias defectuosas
# =============================================================================
print("\n" + "="*50)
print("EJERCICIO 6: Memorias defectuosas")
print("="*50)

# Usando distribución de Poisson
lambda_poisson = 600 * 0.03  # n * p
k_defectuosas = 12
prob_6_poisson = (exp(-lambda_poisson) * (lambda_poisson ** k_defectuosas)) / factorial(k_defectuosas)
print(f"P(12 defectuosas en 600) ≈ {prob_6_poisson:.6f} (aproximación Poisson)")

# Usando distribución binomial (más precisa para este tamaño)
prob_6_binomial = comb(600, 12) * (0.03 ** 12) * (0.97 ** (600 - 12))
print(f"P(12 defectuosas en 600) ≈ {prob_6_binomial:.6f} (distribución binomial)")

# =============================================================================
# GRÁFICOS Y DIAGRAMAS
# =============================================================================
print("\n" + "="*50)
print("GRÁFICOS Y DIAGRAMAS")
print("="*50)

# Gráfico para el Ejercicio 1 - Distribución de estudiantes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Diagrama de pastel para la composición de estudiantes
carreras = ['Electrónica', 'Sistemas', 'Industrial']
estudiantes = [electronica, sistemas, industrial]
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax1.pie(estudiantes, labels=carreras, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Distribución de Estudiantes por Carrera')

# Gráfico de barras para probabilidades del Ejercicio 1 (sin sustitución)
probabilidades = [prob_1a, prob_2a, prob_3a, prob_4a, prob_5a, prob_6a]
etiquetas = ['3 Elect', '3 Sis', '2E-1S', '≥1 Sis', '1 cada', 'Orden E-S-I']
bars = ax2.bar(etiquetas, probabilidades, color='skyblue')
ax2.set_title('Probabilidades - Selección sin Sustitución')
ax2.set_ylabel('Probabilidad')
ax2.tick_params(axis='x', rotation=45)

# Añadir valores en las barras
for bar, prob in zip(bars, probabilidades):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{prob:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Gráfico para el Ejercicio 5 - Distribución binomial
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

# Distribución binomial para el ejercicio 5b
n = 5  # lanzamientos
p = 1/6  # probabilidad de éxito
x = np.arange(0, n+1)
probabilidades_binomial = [comb(n, k) * (p ** k) * ((1-p) ** (n-k)) for k in x]

ax3.bar(x, probabilidades_binomial, color='lightgreen', alpha=0.7)
ax3.set_xlabel('Número de seises')
ax3.set_ylabel('Probabilidad')
ax3.set_title('Distribución Binomial - 5 lanzamientos de dado')
ax3.grid(True, alpha=0.3)

# Resaltar la probabilidad de obtener exactamente 3 seises
ax3.bar(3, probabilidades_binomial[3], color='red', alpha=0.8)
ax3.text(3, probabilidades_binomial[3] + 0.01, f'P(3) = {probabilidades_binomial[3]:.4f}',
         ha='center', va='bottom', fontweight='bold')

# Distribución de Poisson para el ejercicio 6
lambda_val = 18
x_poisson = np.arange(0, 30)
probabilidades_poisson = [exp(-lambda_val) * (lambda_val ** k) / factorial(k) for k in x_poisson]

ax4.bar(x_poisson, probabilidades_poisson, color='lightcoral', alpha=0.7)
ax4.set_xlabel('Número de memorias defectuosas')
ax4.set_ylabel('Probabilidad')
ax4.set_title('Distribución de Poisson - Memorias Defectuosas')
ax4.grid(True, alpha=0.3)

# Resaltar la probabilidad de obtener exactamente 12 defectuosas
ax4.bar(12, probabilidades_poisson[12], color='blue', alpha=0.8)
ax4.text(12, probabilidades_poisson[12] + 0.01, f'P(12) = {probabilidades_poisson[12]:.4f}',
         ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Diagrama de árbol simplificado para el Ejercicio 1
print("\nDIAGRAMA DE ÁRBOL SIMPLIFICADO (Ejercicio 1 - Sin Sustitución)")
print("Primera selección:")
print(f"  - Electrónica: {electronica}/{total_estudiantes} = {electronica/total_estudiantes:.2f}")
print(f"  - Sistemas: {sistemas}/{total_estudiantes} = {sistemas/total_estudiantes:.2f}")
print(f"  - Industrial: {industrial}/{total_estudiantes} = {industrial/total_estudiantes:.2f}")

print("\nEjemplo: Probabilidad de orden E-S-I")
print(f"  1º Electrónica: {electronica}/{total_estudiantes}")
print(f"  2º Sistemas: {sistemas}/{total_estudiantes-1} (quedan {total_estudiantes-1} estudiantes)")
print(f"  3º Industrial: {industrial}/{total_estudiantes-2} (quedan {total_estudiantes-2} estudiantes)")
print(f"  Probabilidad total: ({electronica}/{total_estudiantes}) × ({sistemas}/{total_estudiantes-1}) × ({industrial}/{total_estudiantes-2})")
print(f"  = {prob_6a:.4f}")

print("\n" + "="*60)
print("PROGRAMA EJECUTADO EXITOSAMENTE")
print("="*60)