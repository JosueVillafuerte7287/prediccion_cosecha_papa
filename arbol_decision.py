import csv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def cargar_datos(csv_filename):
    resultados = []

    with open(csv_filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Omitir la primera fila
        for row in csv_reader:
            processed_row = []
            for value in row:
                try:
                    processed_value = float(value)  # Convertir el valor a un número
                    processed_row.append(processed_value)
                except ValueError:
                    pass  # Si no se puede convertir, simplemente omitir ese valor
            resultados.append(processed_row)

    return resultados

def crear_dataframe(resultados):
    return pd.DataFrame(resultados, columns=[f"Feature_{i}" for i in range(len(resultados[0]))])

def calcular_matriz_correlacion(df_resultados):
    return df_resultados.corr()

def dividir_datos_entrenamiento_prueba(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def inicializar_modelo_regresion():
    return LinearRegression()

def entrenar_modelo_regresion(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)

def predecir_modelo_regresion(modelo, X_test):
    return modelo.predict(X_test)

def calcular_error_cuadratico_medio(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)

def visualizar_regresion_lineal(X_train, y_test, y_pred):
    if X_train.shape[1] == 2:
        plt.scatter(X_test.iloc[:, 0], y_test, color='black')
        plt.plot(X_test.iloc[:, 0], y_pred, color='blue', linewidth=3)
        plt.xlabel('Feature_0')
        plt.ylabel('Variable Objetivo')
        plt.title('Regresión Lineal')
        plt.show()

def inicializar_modelo_arbol_decision():
    return DecisionTreeClassifier()

def entrenar_modelo_arbol_decision(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)

def visualizar_arbol_decision(arbol_decision, X_columns):
    plt.figure(figsize=(12, 8))
    plot_tree(arbol_decision, feature_names=X_columns, filled=True, rounded=True)
    plt.show()

def main():
    # Llamar a la función principal aquí
    resultados = cargar_datos('potato-production-districtwise.csv')

    # Crear DataFrame con los resultados
    df_resultados = crear_dataframe(resultados)

    # Crear el mapa de calor solo con los valores numéricos
    matriz_corr = calcular_matriz_correlacion(df_resultados)

    print("Matriz de correlacion: ", matriz_corr)

    # Agregar regresión lineal
    X = df_resultados.iloc[:, :-1]  # Todas las columnas excepto la última como características
    y = df_resultados.iloc[:, -1]   # Última columna como variable objetivo

    # Dividir el conjunto de datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = dividir_datos_entrenamiento_prueba(X, y)

    # Inicializar el modelo de regresión lineal
    modelo_regresion = inicializar_modelo_regresion()

    # Entrenar el modelo
    entrenar_modelo_regresion(modelo_regresion, X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = predecir_modelo_regresion(modelo_regresion, X_test)

    # Calcular el error cuadrático medio (MSE)
    mse = calcular_error_cuadratico_medio(y_test, y_pred)
    print("Error cuadrático medio:", mse)

    # Visualizar la regresión lineal
    visualizar_regresion_lineal(X_train, y_test, y_pred)

    # Agregar árbol de decisión
    arbol_decision = inicializar_modelo_arbol_decision()
    entrenar_modelo_arbol_decision(arbol_decision, X_train, y_train)

    # Visualizar el árbol de decisión
    visualizar_arbol_decision(arbol_decision, X.columns)

if __name__ == "__main__":
    main()
