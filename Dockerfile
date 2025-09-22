# Imagen multi-arch (funciona en M1 y Windows)
FROM python:3.9-bookworm

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

# Instala las dependencias con versiones exactas que funcionan
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Establece el comando predeterminado para ejecutar al iniciar el contenedor
CMD ["tail", "-f", "/dev/null"]