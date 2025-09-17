"""
Demo de la librería MLxtend para Machine Learning
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Configuración de matplotlib
plt.style.use('default')

# 1. ANÁLISIS DE ASOCIACIÓN (MARKET BASKET ANALYSIS)
print("1. ANÁLISIS DE ASOCIACIÓN CON APRIORI")
# Datos de ejemplo: transacciones de un supermercado
transacciones = [
    ['pan', 'leche'],
    ['pan', 'pañales', 'cerveza', 'huevos'],
    ['leche', 'pañales', 'cerveza', 'refresco'],
    ['pan', 'leche', 'pañales', 'cerveza'],
    ['pan', 'leche', 'pañales', 'refresco']
]

# Preprocesamiento con TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transacciones).transform(transacciones)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicación del algoritmo Apriori
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Itemsets frecuentes:")
print(frequent_itemsets)
print("\nReglas de asociación:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 2. CLASIFICACIÓN Y REGIONES DE DECISIÓN
print("\n2. CLASIFICACIÓN Y VISUALIZACIÓN")
# Carga de datos Iris
iris = load_iris()
X = iris.data[:, [0, 2]]  # Solo sepal length y petal length
y = iris.target

# Entrenamiento de modelo
lr = LogisticRegression()
lr.fit(X, y)

# Visualización de regiones de decisión
plt.figure(figsize=(10, 6))
plot_decision_regions(X, y, clf=lr, legend=2)
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Longitud del pétalo (cm)')
plt.title('Regiones de Decisión - Regresión Logística')
plt.show()

# 3. MODELOS DE ENSEMBLE
print("\n3. MODELO DE ENSEMBLE")
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Creación de ensemble
clf1 = LogisticRegression(random_state=1)
clf2 = DecisionTreeClassifier(random_state=1)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2], weights=[1, 1])

# Entrenamiento y evaluación
eclf.fit(X_train_std, y_train)
y_pred = eclf.predict(X_test_std)
print(f'Precisión del ensemble: {accuracy_score(y_test, y_pred):.2%}')

# 4. DESCOMPOSICIÓN SESGO-VARIANZA
print("\n4. ANÁLISIS SESGO-VARIANZA")
X_train_sub = X_train_std[:100]
y_train_sub = y_train[:100]

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
    clf2, X_train_sub, y_train_sub, X_test_std, y_test,
    loss='0-1_loss', random_seed=1)

print(f'Pérdida promedio: {avg_expected_loss:.4f}')
print(f'Sesgo promedio: {avg_bias:.4f}')
print(f'Varianza promedio: {avg_var:.4f}')

# 5. OTRAS FUNCIONALIDADES (Ejemplo de preprocesamiento)
print("\n5. PREPROCESAMIENTO DE DATOS")
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import PrincipalComponentAnalysis

# Estandarización
X_std = standardize(X)
print("Datos estandarizados (primeras 5 filas):")
print(X_std[:5])

# PCA
pca = PrincipalComponentAnalysis(n_components=2)
X_pca = pca.fit(X_std).transform(X_std)
print("\nDatos transformados con PCA (primeras 5 filas):")
print(X_pca[:5])