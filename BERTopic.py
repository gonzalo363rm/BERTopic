import time

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

from multiprocessing import freeze_support

# Timer starts
starttime = time.time()
lasttime = starttime

if __name__ == '__main__':
    freeze_support()

    # Carga el CSV
    data = pd.read_csv("tweets_municipalidad.csv")

    # Prepara el texto
    docs = data["tweet"].astype(str)

    # Fine-tune your topic representations
    representation_model = KeyBERTInspired()

    # Crea el modelo BERTopic
    topic_model = BERTopic(verbose=True, language="spanish", representation_model=representation_model, nr_topics=15)

    # Entrena el modelo
    topics, _ = topic_model.fit_transform(docs)

    # Timer after get topics
    print('time_after_get_topic => ', time.time() - lasttime)
    time_after_get_topic = time.time()

    # Preprocess Documents
    documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
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

    # Extract words in each topic if they are non-empty and exist in the dictionary
    topic_words = []
    for topic in range(len(set(topics))-topic_model._outliers):
        words = list(zip(*topic_model.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)
        topic_words = [words for words in topic_words if len(words) > 0]
    
    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v')
    coherence = coherence_model.get_coherence()

    # Timer after get coherence
    print('time_after_get_coherence => ', time.time() - time_after_get_topic)
    print('total time => ', time.time() - lasttime)

    print(coherence)