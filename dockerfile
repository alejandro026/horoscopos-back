# Usamos una imagen base de Python
FROM python:3.11.4-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos los archivos necesarios al contenedor
COPY requirements.txt /app/
COPY app.py /app/

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto 5000 para Flask
EXPOSE 5000

# Comando para iniciar la aplicación Flask
CMD ["python", "app.py"]
