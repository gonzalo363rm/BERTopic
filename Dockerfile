# Imagen base ligera y multi-arch (funciona en M1 y Windows)
FROM python:3.9-slim-bullseye

# Evita que python guarde archivos .pyc y mejora logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los todos los archivos al contenedor
COPY . .

# Creación de carpetas
RUN mkdir -p own/results/Basic/20NewsGroup \
    own/results/Basic/BBC_News \
    own/results/Basic/Trump \
    own/results/Basic/tweets_municipalidad \
    own/results/Basic/tweets_municipalidad_short \
    own/results/Basic/tweets_preprocessed

# Instala las dependencias
RUN pip install numpy==1.21.6
RUN pip install -e .
RUN pip install bertopic==0.9.4
# Instalamos la ultima version en la que aun existe la función "append" de pandas
RUN pip install pandas==1.5.3

# Establece el comando predeterminado para ejecutar al iniciar el contenedor
CMD ["tail", "-f", "/dev/null"]