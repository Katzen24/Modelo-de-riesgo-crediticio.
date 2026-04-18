# Simulador de Riesgo Crediticio 🏦📊

Este proyecto es una aplicación interactiva de Machine Learning diseñada para predecir la probabilidad de que un cliente incumpla con un préstamo (Default). Utiliza un modelo de clasificación entrenado con datos históricos para evaluar el perfil de riesgo financiero de manera inmediata.

## Tecnologías Utilizadas
* **Python 3.11**: Lenguaje base para el análisis y modelado.
* **Streamlit**: Framework para la creación de la interfaz web interactiva.
* **Scikit-Learn**: Biblioteca utilizada para el entrenamiento del modelo (Random Forest).
* **Pandas & Numpy**: Manipulación y limpieza de datos.
* **Joblib**: Para la persistencia y carga de los modelos entrenados (.pkl).
* **Docker**: Para asegurar la portabilidad y el despliegue consistente.

## Estructura del Proyecto
* `Riesgo_crediticio_app.py`: Aplicación principal de la interfaz de usuario.
* `Riesgo_crediticio.py`: Script de procesamiento y entrenamiento del modelo.
* `modelo_riesgo.pkl`: Modelo entrenado guardado.
* `columnas_modelo.pkl`: Lista de variables necesarias para la predicción.
* `requirements.txt`: Dependencias del proyecto.
* `Dockerfile`: Configuración del contenedor.

## Ejecución con Docker

Para correr este simulador en cualquier sistema sin instalar dependencias manualmente:

1. **Construir la imagen:**
   ```bash
   docker build -t simulador-riesgo .