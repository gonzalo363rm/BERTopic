from evaluation import Trainer, DataLoader
from sentence_transformers import SentenceTransformer

#=========================================
#
# BERTopic__20NEWSGROUP
#
#=========================================

# Prepare data
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
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/20NewsGroup/bertopic_{i+1}")

# #=========================================
# #
# # BERTopic__BBC_news
# #
# #=========================================

# Prepare data
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
#         "nr_topics": [(i+1)*5 for i in range(10)],
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

# Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "../../datasets/trump", True
# Si no esta descargado el dataset de trump, descomentar la siguiente linea:
# dataloader = DataLoader(dataset).prepare_docs(save="../../datasets/trump.txt").preprocess_octis(output_folder="../../datasets/trump")

# Prepare data
# data_loader = DataLoader(dataset)
# _, timestamps = data_loader.load_docs()
# data = data_loader.load_octis(custom)
# data = [" ".join(words) for words in data.get_corpus()]

# Extract embeddings
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
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/Trump/bertopic_{i+1}")

# =========================================
# 
# BERTopic_tweets_municipalidad
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
dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_municipalidad", documents_path="/app/own/datasets/tweets_municipalidad.txt")

# Prepare data
data = dataloader.load_octis(custom)
data = [" ".join(words) for words in data.get_corpus()]

# Extract embeddings
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(data, show_progress_bar=True)

params = {
        "embedding_model": "all-mpnet-base-v2",
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
    results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/tweets_municipalidad/bertopic_{i+1}")

#=========================================
#
# BERTopic_senadores with preprocess
#
#=========================================

# Abre el archivo en modo de lectura
# with open('/app/own/datasets/senadores_preprocessed.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     senadores_preprocessed = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/senadores_preprocessed", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/senadores_preprocessed.txt", docs=senadores_preprocessed)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/senadores_preprocessed", documents_path="/app/own/datasets/senadores_preprocessed.txt")

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
#     results = trainer.train(save=f"/app/own/models/BERTopic/results/Basic/senadores_preprocessed/bertopic_{i+1}")