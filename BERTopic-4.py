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
    from multiprocessing import freeze_support
    
    # Specify the path to your file
    file_path = './file.txt'

    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # Creating a dataframe from the data imported 
    full_train = pd.DataFrame({'text': lines})
    documents = full_train

    filtered_text = lines

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=10
    )

    topics, probabilities = topic_model.fit_transform(filtered_text)
    topic_model.visualize_topics()
    topic_model.visualize_barchart()

    documents = pd.DataFrame({"Document": filtered_text,
                              "ID": range(len(filtered_text)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    words = vectorizer_model.get_feature_names_out()
    tokens = [vectorizer_model.build_analyzer()(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                   for topic in range(len(set(topics))-1)]

    freeze_support()
    main()