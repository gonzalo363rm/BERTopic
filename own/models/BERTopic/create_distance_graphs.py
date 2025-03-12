import os
from bertopic import BERTopic

# NOTA: NO SE USA POR POSIBLE PROBLEMA DE MEMORIA, NO SE GUARDAN LOS MODELOS SINO QUE SE GRAFICA DIRECTAMENTE EN EL ENTRENAMIENTO (evaluation/evaluation.py)

# Ruta al directorio que contiene los modelos entrenados
models_dir = "/app/own/models/BERTopic/trained_models"

# Ruta al directorio donde se guardarán las gráficas
graphs_dir = "/app/own/graphs/distance"

# Crear el directorio de gráficas si no existe
os.makedirs(graphs_dir, exist_ok=True)

# Recorrer todos los archivos en el directorio de modelos
for filename in os.listdir(models_dir):
    # Construir la ruta completa al archivo
    model_path = os.path.join(models_dir, filename)

    # Verificar si es un archivo (y no un subdirectorio)
    if os.path.isfile(model_path):
        try:
            # Cargar el modelo
            loaded_model = BERTopic.load(model_path)

            # Generar la gráfica de los temas
            fig = loaded_model.visualize_topics()

            # Crear un nombre de archivo para la gráfica (igual al nombre del modelo)
            graph_filename = f"{os.path.splitext(filename)[0]}.html"  # Mismo nombre, extensión .html
            graph_path = os.path.join(graphs_dir, graph_filename)

            # Guardar la gráfica en un archivo HTML
            fig.write_html(graph_path)
            print(f"Gráfica guardada en: {graph_path}")

        except Exception as e:
            print(f"Error al procesar el archivo {filename}: {e}")