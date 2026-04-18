FROM python:3.11-slim

# Instalamos dependencias para gráficas y compilación
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos e instalamos librerías
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el proyecto (incluyendo los .pkl)
COPY . .

# Exponemos el puerto de Streamlit
EXPOSE 8501

# Ejecutamos la app
CMD ["streamlit", "run", "Riesgo_crediticio_app.py", "--server.port=8501", "--server.address=0.0.0.0"]