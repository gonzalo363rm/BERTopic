from evaluation import Trainer, DataLoader
from sentence_transformers import SentenceTransformer

# #=========================================
# #
# # LDA__20NEWSGROUP
# #
# #=========================================

# Prepare data
# dataset, custom = "20NewsGroup", False
# for i, random_state in enumerate([0, 21, 42]):
#     params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": random_state}

#     trainer = Trainer(dataset=dataset,
#                       model_name="LDA",
#                       params=params,
#                       custom_dataset=custom,
#                       verbose=True)
#     results = trainer.train(save=f"/app/own/models/LDA/results/Basic/20NewsGroup/lda_{i+1}")

# #=========================================
# #
# # LDA__BBC_news
# #
# #=========================================

# Prepare data
# dataset, custom = "BBC_News", False
# for i, random_state in enumerate([0, 21, 42]):
#     params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": random_state}

#     trainer = Trainer(dataset=dataset,
#                       model_name="LDA",
#                       params=params,
#                       custom_dataset=custom,
#                       verbose=True)
#     results = trainer.train(save=f"/app/own/models/LDA/results/Basic/BBC_News/lda_{i+1}")

# #=========================================
# #
# # LDA__trump
# #
# #=========================================

# Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "../../datasets/trump", True
# Si no esta descargado el dataset de trump, descomentar la siguiente linea:
# dataloader = DataLoader(dataset).prepare_docs(save="../../datasets/trump.txt").preprocess_octis(output_folder="../../datasets/trump")

# for i, random_state in enumerate([0, 21, 42]):
#     params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": random_state}

#     trainer = Trainer(dataset=dataset,
#                       model_name="LDA",
#                       params=params,
#                       custom_dataset=custom,
#                       verbose=True)
#     results = trainer.train(save=f"/app/own/models/LDA/results/Basic/Trump/lda_{i+1}")

# =========================================
# 
# LDA_tweets_municipalidad
# 
# =========================================

# Abre el archivo en modo de lectura
with open('/app/own/datasets/tweets_municipalidad.txt', 'r', encoding='utf-8') as file:
    # Lee las líneas
    tweets_municipalidad = file.readlines()

# Prepare the documents and save them in an OCTIS-based format
dataset, custom = "/app/own/datasets/tweets_municipalidad", True
dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_municipalidad.txt", docs=tweets_municipalidad)
# Esto debe correrse al menos una vez para crear la carpeta tweets_municipalidad con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_municipalidad", documents_path="/app/own/datasets/tweets_municipalidad.txt")

for i, random_state in enumerate([0, 21, 42]):
    params = {"num_topics": [(i+1)*5 for i in range(10)], "random_state": random_state}
    params = {"num_topics": [(i+1)*5 for i in range(10)], "random_state": random_state, "alpha": 4.38393917610078, "eta": 4.9725466249069346} # Parametros optimizados
    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    # results = trainer.train(save=f"/app/own/models/LDA/results/Basic/tweets_municipalidad/lda_{i+1}")
    results = trainer.train(save=f"/app/own/models/LDA/results/Basic/tweets_municipalidad/lda_optimizado_{i+1}")

#=========================================
#
# LDA_senadores
#
#=========================================

# Abre el archivo en modo de lectura
# with open('/app/own/datasets/senadores.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     senadores = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/senadores", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/senadores.txt", docs=senadores)
# # Esto debe correrse al menos una vez para crear la carpeta senadores con los archivos (corpus.tsv, indexes.txt, metadata.json y vocabulary.txt)
# # dataloader.preprocess_octis(output_folder="/app/own/datasets/senadores", documents_path="/app/own/datasets/senadores.txt")

# # Prepare data
# data = dataloader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# # Extract embeddings
# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data, show_progress_bar=True)

# for i, random_state in enumerate([0, 21, 42]):
#     params = {"num_topics": [(i+1)*5 for i in range(10)], "random_state": random_state}
#     trainer = Trainer(dataset=dataset,
#                       model_name="LDA",
#                       params=params,
#                       custom_dataset=custom,
#                       verbose=True)
#     results = trainer.train(save=f"/app/own/models/LDA/results/Basic/senadores/lda_{i+1}")