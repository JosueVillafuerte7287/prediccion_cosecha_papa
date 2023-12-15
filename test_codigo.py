import unittest
import pandas as pd
from arbol_decision import (cargar_datos, crear_dataframe, calcular_matriz_correlacion,
                    dividir_datos_entrenamiento_prueba, inicializar_modelo_regresion,
                    entrenar_modelo_regresion, predecir_modelo_regresion,
                    calcular_error_cuadratico_medio, visualizar_regresion_lineal,
                    inicializar_modelo_arbol_decision, entrenar_modelo_arbol_decision,
                    visualizar_arbol_decision)

class TestCodigo(unittest.TestCase):
    def setUp(self):
        # Puedes inicializar datos de prueba aquí si es necesario
        self.csv_filename = 'potato-production-districtwise.csv'

    def test_cargar_datos(self):
        resultados = cargar_datos(self.csv_filename)
        # Agrega aserciones para verificar si la carga de datos es correcta
        self.assertIsInstance(resultados, list)
        self.assertGreater(len(resultados), 0)

    def test_crear_dataframe(self):
        # Crea datos de prueba para el DataFrame
        datos_prueba = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        df_resultados = crear_dataframe(datos_prueba)
        # Agrega aserciones para verificar si el DataFrame se crea correctamente
        self.assertIsInstance(df_resultados, pd.DataFrame)
        self.assertEqual(df_resultados.shape, (len(datos_prueba), len(datos_prueba[0])))

    # Agrega más pruebas para las demás funciones

if __name__ == '__main__':
    unittest.main()
