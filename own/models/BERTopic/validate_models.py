from evaluation import Trainer, DataLoader
from sentence_transformers import SentenceTransformer

#=========================================
#
# BERTopic__20NEWSGROUP
#
#=========================================

# # Prepare data
# dataset, custom = "20NewsGroup", False
# data_loader = DataLoader(dataset)
# _, timestamps = data_loader.load_docs()
# data = data_loader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data, show_progress_bar=True)

# params = {
#         "embedding_model": "all-mpnet-base-v2",
#         "nr_topics": [(i+1)*10 for i in range(5)],
#         "min_topic_size": 15,
#         "verbose": True
#     }

# for i in range(3):
#     trainer = Trainer(dataset=dataset,
#                     model_name="BERTopic",
#                     params=params,
#                     bt_embeddings=embeddings,
#                     custom_dataset=custom,
#                     verbose=True)
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/20NewsGroup/bertopic_{i+1}")

# #=========================================
# #
# # BERTopic__BBC_news
# #
# #=========================================

# # Prepare data
# dataset, custom = "BBC_News", False
# data_loader = DataLoader(dataset)
# _, timestamps = data_loader.load_docs()
# data = data_loader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data, show_progress_bar=True)

# params = {
#         "embedding_model": "all-mpnet-base-v2",
#         "nr_topics": [(i+1)*10 for i in range(5)],
#         # "min_topic_size": 15, Segun el autor para este data set no va
#         "verbose": True
#     }

# for i in range(3):
#     trainer = Trainer(dataset=dataset,
#                     model_name="BERTopic",
#                     params=params,
#                     bt_embeddings=embeddings,
#                     custom_dataset=custom,
#                     verbose=True)
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/BBC_News/bertopic_{i+1}")

# #=========================================
# #
# # BERTopic__trump
# #
# #=========================================

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "../../datasets/trump", True
# # Si no esta descargado el dataset de trump, descomentar la siguiente linea:
# # dataloader = DataLoader(dataset).prepare_docs(save="../../datasets/trump.txt").preprocess_octis(output_folder="../../datasets/trump")

# # Prepare data
# data_loader = DataLoader(dataset)
# _, timestamps = data_loader.load_docs()
# data = data_loader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data, show_progress_bar=True)

# params = {
#         "embedding_model": "all-mpnet-base-v2",
#         "nr_topics": [(i+1)*10 for i in range(5)],
#         "min_topic_size": 15,
#         "verbose": True
#     }

# for i in range(3):
#     trainer = Trainer(dataset=dataset,
#                     model_name="BERTopic",
#                     params=params,
#                     bt_embeddings=embeddings,
#                     custom_dataset=custom,
#                     verbose=True)
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/Trump/bertopic_{i+1}")

# =========================================
# 
# BERTopic_tweets_municipalidad
# 
# =========================================

# Abre el archivo en modo de lectura
# with open('/app/own/datasets/tweets_municipalidad.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     tweets_municipalidad = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/tweets_municipalidad", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_municipalidad.txt", docs=tweets_municipalidad)
# # Esto debe correrse al menos una vez para crear la carpeta tweets_municipalidad con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_municipalidad", documents_path="/app/own/datasets/tweets_municipalidad.txt")

# # Prepare data
# data = dataloader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data, show_progress_bar=True)

# params = {
#         "embedding_model": "all-mpnet-base-v2",
#         "nr_topics": [(i+1)*5 for i in range(10)],
#         "min_topic_size": 15,
#         "verbose": True
#     }

# for i in range(3):
#     trainer = Trainer(dataset=dataset,
#                     model_name="BERTopic",
#                     params=params,
#                     bt_embeddings=embeddings,
#                     custom_dataset=custom,
#                     verbose=True)
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/tweets_municipalidad/bertopic_{i+1}")

# =========================================

# BERTopic_senadores

# =========================================

# Abre el archivo en modo de lectura
# with open('/app/own/datasets/senadores.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     senadores = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/senadores", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/senadores.txt", docs=senadores)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/senadores", documents_path="/app/own/datasets/senadores.txt")

# # Prepare data
# data = dataloader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data, show_progress_bar=True)

# params = {
#         "embedding_model": "all-mpnet-base-v2",
#         "nr_topics": [(i+1)*5 for i in range(10)],
#         "min_topic_size": 15,
#         "verbose": True
#     }

# for i in range(3):
#     trainer = Trainer(dataset=dataset,
#                     model_name="BERTopic",
#                     params=params,
#                     bt_embeddings=embeddings,
#                     custom_dataset=custom,
#                     verbose=True)
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/senadores/bertopic_{i+1}")

# =========================================
# 
# BERTopic_tweets_preprocessed con Dask (optimizado para 8GB RAM)
# 
# =========================================

import dask.dataframe as dd
import gc
import numpy as np

# Cargar dataset como Dask DataFrame con bloques pequeños
tweets_ddf = dd.read_csv('/app/own/datasets/tweets_preprocessed.txt', 
                         header=None, 
                         blocksize='32MB')  # Bloques de 32MB para 8GB RAM

# Convertir a lista de textos por chunks
def process_chunk(chunk):
    return chunk[0].tolist()

# Procesar por chunks conservadores para 8GB RAM
chunk_size = 25000  # 25k tweets por chunk (más conservador)
chunks = []
total_tweets = len(tweets_ddf)

print(f"Total de tweets a procesar: {total_tweets:,}")
print(f"Procesando en chunks de {chunk_size:,} tweets")

# Usar map_partitions para procesar por chunks
def process_partition(partition):
    return partition[0].tolist()

# Procesar cada partición de Dask
partitions = tweets_ddf.map_partitions(process_partition).compute()
chunks = [part for part in partitions if part]  # Filtrar particiones vacías

print(f"  - Chunks procesados: {len(chunks)} particiones")
print(f"  - Total de tweets en chunks: {sum(len(chunk) for chunk in chunks):,}")

# Prepare the documents and save them in an OCTIS-based format
dataset, custom = "/app/own/datasets/tweets_preprocessed", True

# Usar el primer chunk para preparar la estructura OCTIS
first_chunk = chunks[0] if chunks else []
dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_preprocessed.txt", docs=first_chunk)
# Esto debe correrse al menos una vez para crear la carpeta tweets_preprocessed con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_preprocessed", documents_path="/app/own/datasets/tweets_preprocessed.txt")

# Prepare data
data = dataloader.load_octis(custom)
data = [" ".join(words) for words in data.get_corpus()]

# Procesar embeddings por chunks y guardar en disco (optimizado para 8GB RAM)
model = SentenceTransformer("all-mpnet-base-v2")
embedding_files = []

print("\nProcesando embeddings por chunks y guardando en disco...")

for i, chunk_texts in enumerate(chunks):
    print(f"Embeddings chunk {i+1}/{len(chunks)} ({len(chunk_texts):,} tweets)")
    
    # Procesar embeddings del chunk con batch size muy pequeño para 8GB RAM
    chunk_embeddings = model.encode(chunk_texts, 
                                   show_progress_bar=True,
                                   batch_size=8,  # Batch muy pequeño para 8GB RAM
                                   device='cpu')    # Usar CPU para evitar problemas de memoria GPU
    
    # Guardar embeddings del chunk en disco
    embedding_file = f"/tmp/embeddings_chunk_{i}.npy"
    np.save(embedding_file, chunk_embeddings)
    embedding_files.append(embedding_file)
    
    # Limpiar memoria inmediatamente
    del chunk_texts, chunk_embeddings
    gc.collect()
    
    print(f"  - Embeddings del chunk {i+1} guardados en {embedding_file}")
    print(f"  - Memoria liberada")

print(f"\nTotal de archivos de embeddings: {len(embedding_files)}")

# Cargar y combinar embeddings por chunks para evitar saturar memoria
print("\nCargando embeddings por chunks para entrenamiento...")
embeddings = None

for i, embedding_file in enumerate(embedding_files):
    print(f"Cargando embeddings chunk {i+1}/{len(embedding_files)}")
    
    chunk_embeddings = np.load(embedding_file)
    
    if embeddings is None:
        embeddings = chunk_embeddings
    else:
        embeddings = np.vstack([embeddings, chunk_embeddings])
    
    # Limpiar memoria del chunk cargado
    del chunk_embeddings
    gc.collect()
    
    print(f"  - Embeddings acumulados: {embeddings.shape}")

print(f"Embeddings finales: {embeddings.shape}")

# Limpiar archivos temporales
for embedding_file in embedding_files:
    try:
        import os
        os.remove(embedding_file)
        print(f"Archivo temporal eliminado: {embedding_file}")
    except:
        pass

print("Memoria optimizada, continuando con entrenamiento...")

params = {
        "embedding_model": "all-mpnet-base-v2",
        "nr_topics": [(i+1)*5 for i in range(10)],
        "min_topic_size": 15,
        "verbose": True
    }

for i in range(3):
    print(f"\nEntrenando modelo {i+1}/3...")
    trainer = Trainer(dataset=dataset,
                    model_name="BERTopic",
                    params=params,
                    bt_embeddings=embeddings,
                    custom_dataset=custom,
                    verbose=True)
    results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/tweets_preprocessed/bertopic_{i+1}")
    print(f"Modelo {i+1} entrenado y guardado")