from evaluation import Trainer, DataLoader
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
import numpy as np

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
# BERTopic_tweets_preprocessed
# 
# =========================================

# Abre el archivo en modo de lectura
# with open('/app/own/datasets/tweets_preprocessed.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     tweets_preprocessed = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/tweets_preprocessed", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_preprocessed.txt", docs=tweets_preprocessed)
# # Esto debe correrse al menos una vez para crear la carpeta tweets_preprocessed con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_preprocessed", documents_path="/app/own/datasets/tweets_preprocessed.txt")

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
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/tweets_preprocessed/bertopic_{i+1}")

# =========================================
# 
# BERTopic_tweets_preprocessed usando BETO como embedding
# 
# =========================================

# # Abre el archivo en modo de lectura
# with open('/app/own/datasets/tweets_preprocessed.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     tweets_preprocessed_beto = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/tweets_preprocessed", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_preprocessed.txt", docs=tweets_preprocessed)
# # Esto debe correrse al menos una vez para crear la carpeta tweets_preprocessed con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_preprocessed", documents_path="/app/own/datasets/tweets_preprocessed.txt")

# # Prepare data
# data = dataloader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# embedding_model = pipeline("feature-extraction", model="dccuchile/bert-base-spanish-wwm-uncased")

# # Calcular embeddings MANUALMENTE una sola vez
# print("Calculando embeddings...")
# embeddings = []

# # Procesar por lotes pequeños para mejor rendimiento
# batch_size = 8
# for i in range(0, len(data), batch_size):
#     if i % 1000 == 0:
#         print(f"Procesando lote {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
    
#     batch = data[i:i+batch_size]
#     batch_embeddings = embedding_model(batch, return_tensors="pt")
    
#     # Tomar el promedio de todos los tokens para cada documento
#     for j in range(len(batch)):
#         embeddings.append(batch_embeddings[j].mean(dim=1).squeeze().detach().numpy())

# embeddings = np.array(embeddings)
# # print(f"Embeddings calculados: {embeddings.shape}")

# params = {
#         "embedding_model": embedding_model,
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
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/tweets_preprocessed_beto/bertopic_{i+1}")

    
# =========================================
# 
# BERTopic_tweets_preprocessed usando hiiamsid/sentence_similarity_spanish_es como embedding
# 
# =========================================

# Abre el archivo en modo de lectura
with open('/app/own/datasets/tweets_preprocessed.txt', 'r', encoding='utf-8') as file:
    # Lee las líneas
    tweets_preprocessed = file.readlines()

# Prepare the documents and save them in an OCTIS-based format
dataset, custom = "/app/own/datasets/tweets_preprocessed", True
dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_preprocessed.txt", docs=tweets_preprocessed)
# Esto debe correrse al menos una vez para crear la carpeta tweets_preprocessed con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_preprocessed", documents_path="/app/own/datasets/tweets_preprocessed.txt")

# Prepare data
data = dataloader.load_octis(custom)
data = [" ".join(words) for words in data.get_corpus()]

# Extract embeddings
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
embeddings = model.encode(data, show_progress_bar=True)

params = {
        "embedding_model": embedding_model,
        "nr_topics": [(i+1)*5 for i in range(10)],
        "min_topic_size": 15,
        "verbose": True
    }

for i in range(3):
    trainer = Trainer(dataset=dataset,
                    model_name="BERTopic",
                    params=params,
                    bt_embeddings=embeddings,
                    custom_dataset=custom,
                    verbose=True)
    results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/tweets_preprocessed_hiiamsid/bertopic_{i+1}")