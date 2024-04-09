import nltk
import gensim
import gensim.corpora as corpora
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from sklearn.datasets import fetch_20newsgroups

from multiprocessing import freeze_support

import pickle

# dataset = fetch_20newsgroups(subset='train')['data']
# print(len(dataset)) #the length of the data
# print(type(dataset)) # the type of variable the data is stored in 
# print(dataset[:1]) # the first instance of the content within the data
# print(dataset[:1])

# Specify the path to your file
file_path = './file.txt'

with open(file_path, 'r') as file:
    lines = file.read().splitlines()

# #Creating a dataframe from the data imported 
full_train = pd.DataFrame({'text': lines})
# full_train = pd.DataFrame()
# full_train['text'] = dataset
# full_train['text'] = full_train['text'].fillna('').astype(str) #removing any nan type objects
full_train.head()
documents = full_train
# documents

# #If the following packages are not already downloaded, the following lines are needed 
# #nltk.download('wordnet')
# #nltk.download('omw-1.4')
# #nltk.download('punkt')

filtered_text = lines
# lemmatizer = WordNetLemmatizer()

# for w in dataset:
#   filtered_text.append(lemmatizer.lemmatize(w))
# # print(filtered_text[:1])

# # Step 2.1 - Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Step 2.2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# # Step 2.3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# # Step 2.4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# # Step 2.5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

topic_model = BERTopic(
  embedding_model=embedding_model,    # Step 1 - Extract embeddings
  umap_model=umap_model,              # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
#   diversity=0.5,
  nr_topics=10                        # Step 6 - Diversify topic words
)

topics, probabilities = topic_model.fit_transform(filtered_text)
topic_model.visualize_topics()
topic_model.visualize_barchart()

documents = pd.DataFrame({"Document": filtered_text,
                          "ID": range(len(filtered_text)),
                          "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

# Extract vectorizer and analyzer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names_out()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

# Guardamos archivos
with open('words.pkl', "wb") as file:
    pickle.dump(words, file)

with open('tokens.pkl', "wb") as file:
    pickle.dump(tokens, file)

with open('dictionary.pkl', "wb") as file:
    pickle.dump(dictionary, file)

with open('corpus.pkl', "wb") as file:
    pickle.dump(corpus, file)

with open('topic_words.pkl', "wb") as file:
    pickle.dump(topic_words, file)
# Terminamos el guardado

# # Leemos
# with open('./words.pkl', 'rb') as archivo:
#     words = pickle.load(archivo)

# with open('./tokens.pkl', 'rb') as archivo:
#     tokens = pickle.load(archivo)

# with open('./dictionary.pkl', 'rb') as archivo:
#     dictionary = pickle.load(archivo)

# with open('./corpus.pkl', 'rb') as archivo:
#     corpus = pickle.load(archivo)

# with open('./topic_words.pkl', 'rb') as archivo:
#     topic_words = pickle.load(archivo)
# # Terminamos la lectura

def main():
    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence='c_v')
    coherence = coherence_model.get_coherence()

    print(coherence)

if __name__ == '__main__':
    freeze_support()
    main()