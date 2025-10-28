# Imagen base
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . .

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto Flask
EXPOSE 3100

# Comando de inicio
CMD ["gunicorn", "-b", "0.0.0.0:3100", "app:app"]
