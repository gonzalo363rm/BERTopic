from evaluation import Trainer, DataLoader
from sentence_transformers import SentenceTransformer

# #=========================================
# #
# # BERTopic__20NEWSGROUP
# #
# #=========================================

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
#     results = trainer.train(save=f"/app/own/results/Basic/20NewsGroup/bertopic_{i+1}")

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
#     results = trainer.train(save=f"/app/own/results/Basic/BBC_News/bertopic_{i+1}")

# #=========================================
# #
# # BERTopic__trump
# #
# #=========================================

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "trump", True
# dataloader = DataLoader(dataset).prepare_docs(save="trump.txt").preprocess_octis(output_folder="trump")

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
#     results = trainer.train(save=f"/app/own/results/Basic/Trump/bertopic_{i+1}")

#=========================================
#
# tweets_municipalidad CSV to TXT and preprocessing
#
#=========================================

# import csv
# from spanish_nlp import preprocess
# import nltk
# nltk.data.path.append(r'C:\Users\gonza\AppData\Roaming\nltk_data')

# input_file = 'own/datasets/tweets_municipalidad_short.csv'
# output_file = 'own/datasets/tweets_municipalidad_short.txt'

# sp = preprocess.SpanishPreprocess(
#     lower=True,
#     remove_url=True,
#     remove_hashtags=False,
#     split_hashtags=True,
#     normalize_breaklines=True,
#     remove_emoticons=False,
#     remove_emojis=False,
#     convert_emoticons=False,
#     convert_emojis=False,
#     normalize_inclusive_language=True,
#     reduce_spam=True,
#     remove_vowels_accents=True,
#     remove_multiple_spaces=True,
#     remove_punctuation=True,
#     remove_unprintable=True,
#     remove_numbers=True,
#     remove_stopwords=True,
#     stopwords_list='nltk', #'default', 'extended', 'nltk', 'spacy'
#     lemmatize=True,
#     stem=False,
#     remove_html_tags=True,
# )

# # Abrimos el archivo CSV y el TXT
# with open(input_file, mode='r', encoding='utf-8') as csv_file, open(output_file, mode='w', encoding='utf-8') as txt_file:
#     csv_reader = csv.reader(csv_file)
    
#     # Saltamos la cabecera
#     next(csv_reader)
    
#     for row in csv_reader:
#         tweet = row[1]
#         # Preprocesamos y eliminamos los saltos de linea
#         preprocessed_tweet = sp.transform(tweet, debug=False).strip()
#         if preprocessed_tweet:
#             print(preprocessed_tweet)
#             txt_file.write(preprocessed_tweet + '\n')

#=========================================
#
# BERTopic_tweets_municipalidad with spanish corpus preprocess
#
#=========================================

# # Abre el archivo en modo de lectura
# with open('/app/own/datasets/tweets_municipalidad.txt', 'r', encoding='utf-8') as file:
#     # Lee las líneas
#     tweets_municipalidad = file.readlines()

# # Prepare the documents and save them in an OCTIS-based format
# dataset, custom = "/app/own/datasets/tweets_municipalidad", True
# dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_municipalidad.txt", docs=tweets_municipalidad)
# dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_municipalidad", documents_path="/app/own/datasets/tweets_municipalidad.txt")

# # Prepare data
# data = dataloader.load_octis(custom)
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
#     results = trainer.train(save=f"/app/own/results/Basic/tweets_municipalidad/bertopic_{i+1}")

#=========================================
#
# BERTopic_tweets_municipalidad_short with spanish corpus preprocess
#
#=========================================

# Abre el archivo en modo de lectura
with open('/app/own/datasets/tweets_municipalidad_short.txt', 'r', encoding='utf-8') as file:
    # Lee las líneas
    tweets_municipalidad_short = file.readlines()

# Prepare the documents and save them in an OCTIS-based format
dataset, custom = "/app/own/datasets/tweets_municipalidad_short", True
dataloader = DataLoader(dataset).prepare_docs(save="/app/own/datasets/tweets_municipalidad_short.txt", docs=tweets_municipalidad_short)
dataloader.preprocess_octis(output_folder="/app/own/datasets/tweets_municipalidad_short", documents_path="/app/own/datasets/tweets_municipalidad_short.txt")

# Prepare data
data = dataloader.load_octis(custom)
data = [" ".join(words) for words in data.get_corpus()]

# Extract embeddings
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(data, show_progress_bar=True)

params = {
        "embedding_model": "all-mpnet-base-v2",
        "nr_topics": [(i+1)*10 for i in range(5)],
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
    results = trainer.train(save=f"/app/own/results/Basic/tweets_municipalidad_short/bertopic_{i+1}")