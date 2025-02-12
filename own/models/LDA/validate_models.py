from evaluation import Trainer, DataLoader
from sentence_transformers import SentenceTransformer

# #=========================================
# #
# # BERTopic__20NEWSGROUP
# #
# #=========================================

# Prepare data
dataset, custom = "20NewsGroup", False
for i, random_state in enumerate([0, 21, 42]):
    params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": random_state}

    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    results = trainer.train(save=f"/app/own/models/LDA/results/Basic/20NewsGroup/lda_{i+1}")

# #=========================================
# #
# # BERTopic__BBC_news
# #
# #=========================================

# Prepare data
dataset, custom = "BBC_News", False
for i, random_state in enumerate([0, 21, 42]):
    params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": random_state}

    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    results = trainer.train(save=f"/app/own/models/LDA/results/Basic/BBC_News/lda_{i+1}")

# #=========================================
# #
# # BERTopic__trump
# #
# #=========================================

# Prepare the documents and save them in an OCTIS-based format
dataset, custom = "../../datasets/trump", True
# Si no esta descargado el dataset de trump, descomentar la siguiente linea:
# dataloader = DataLoader(dataset).prepare_docs(save="../../datasets/trump.txt").preprocess_octis(output_folder="../../datasets/trump")

for i, random_state in enumerate([0, 21, 42]):
    params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": random_state}

    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    results = trainer.train(save=f"/models/LDA/results/Basic/trump/lda_{i+1}")