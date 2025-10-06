import json
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

model_name = "NMF"
dataset = "tweets_preprocessed_200k"

# Path to the folder containing JSON files
folder_path = os.path.join("models", model_name.lower(), "results", "Basic", dataset)

# Path to save the graph
output_folder = "graphs"
os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe

# List of JSON files
json_files = [f"{model_name.lower()}_1.json", f"{model_name.lower()}_2.json", f"{model_name.lower()}_3.json"]

# Diccionarios para almacenar métricas agrupadas por número de tópicos
cv_scores = defaultdict(list)
npmi_scores = defaultdict(list)
diversity_scores = defaultdict(list)

# Leer y extraer métricas de cada archivo JSON
for file in json_files:
    full_path = os.path.join(folder_path, file)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        continue
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraer métricas para cada resultado en el archivo
        for result in data:
            # Extraer el número de tópicos (maneja tanto "nr_topics" como "num_topics")
            params = result.get("Params", {})
            num_topics = params.get("nr_topics") or params.get("num_topics")
            
            if num_topics is None:
                print(f"Advertencia: No se encontró el número de tópicos en {file}")
                continue  # Si no hay número de tópicos, saltar este resultado
            
            # Extraer las métricas
            scores = result.get("Scores", {})
            cv = scores.get("cv")
            npmi = scores.get("npmi")
            diversity = scores.get("diversity")
            
            if cv is None or npmi is None or diversity is None:
                print(f"Advertencia: Faltan métricas en {file}")
                continue  # Si faltan métricas, saltar este resultado
            
            # Agregar métricas a los diccionarios
            cv_scores[num_topics].append(cv)
            npmi_scores[num_topics].append(npmi)
            diversity_scores[num_topics].append(diversity)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Calcular promedios para cada número de tópicos
num_topics_list = sorted(cv_scores.keys())  # Lista ordenada de números de tópicos
cv_avg = [np.mean(cv_scores[num]) for num in num_topics_list]
npmi_avg = [np.mean(npmi_scores[num]) for num in num_topics_list]
diversity_avg = [np.mean(diversity_scores[num]) for num in num_topics_list]

# Verificar los datos
print(f"Número de tópicos: {num_topics_list}")
print(f"CV promedios: {cv_avg}")
print(f"NPMI promedios: {npmi_avg}")
print(f"Diversity promedios: {diversity_avg}")

# Graficar las métricas con ejes invertidos
y = np.arange(len(num_topics_list))  # Usar un índice numérico para el eje Y
height = 0.3  # Aumentar la altura de las barras para hacerlas más gruesas

fig, ax = plt.subplots(figsize=(10, 8))  # Ajustar el tamaño de la figura

# Graficar cada métrica
ax.barh(y - height, cv_avg, height, label='Coherencia (CV)')
ax.barh(y, npmi_avg, height, label='Coherencia (NPMI)')
ax.barh(y + height, diversity_avg, height, label='Diversidad')

# Añadir etiquetas, título y leyenda
ax.set_ylabel('Número de tópicos', fontsize=12)  # Aumentar el tamaño de la fuente
ax.set_xlabel('Puntuación', fontsize=12)  # Aumentar el tamaño de la fuente
ax.set_title(f'{model_name} Resumen de métricas', fontsize=14)  # Aumentar el tamaño de la fuente
ax.set_yticks(y)
ax.set_yticklabels(num_topics_list, fontsize=10)  # Usar num_topics_list como etiquetas y aumentar la fuente
ax.tick_params(axis='x', labelsize=10)  # Aumentar el tamaño de la fuente en el eje X
ax.legend(fontsize=10)  # Aumentar el tamaño de la fuente de la leyenda

plt.tight_layout()

# Guardar la gráfica en un archivo
output_path = os.path.join(output_folder, f"{model_name.lower()}_{dataset}_metrics_inverted.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gráfica guardada en: {output_path}")