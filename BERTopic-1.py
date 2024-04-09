import sys
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
# from sklearn.datasets import fetch_20newsgroups

# Importaciones para el preprocesamiento
from nltk.corpus import stopwords
import spacy
from spacy.lang.es import Spanish
import nltk
# nltk.download('stopwords')
from unidecode import unidecode

# Nuestros Tuits
# docs = pd.read_csv('C:/Users/gonza/Desktop/J/wFacu/Tesina/BERTopic/tweets_municipalidad.csv')
docs = pd.read_csv('C:/Users/gonza/Desktop/J/wFacu/Tesina/BERTopic/tweets_municipalidad-short.csv')
docs = docs['tweet'].astype(str)
print(docs)

# Preprocesamiento
nlp = spacy.load('es_core_news_sm') # Spanish
spanish_stopwords = set(stopwords.words('spanish'))

def preprocess(text):
    result = ''
    doc = nlp(text)
    for token in doc:
        # if token.is_alpha and token.text.lower() not in spanish_stopwords and len(token.text) > 3:
            token_text = unidecode(token.text.lower()) # Aplicar unidecode al texto del token
            result = result + ' ' + token_text if result else token_text
    return result


processed_data = docs

# Quitamos arrays vacios
processed_data_without_empty_lists = [item for item in processed_data if item]
# print(processed_data_without_empty_lists)

# print(processed_data_without_empty_lists)
# sys.exit()

# Inicializamos el modelo


# Fine-tune your topic representations
representation_model = KeyBERTInspired()
topic_model = BERTopic(language="spanish", representation_model=representation_model)

# Corremos el modelo
topics, probs = topic_model.fit_transform(processed_data_without_empty_lists)
# df = pd.DataFrame({ "Document": processed_data_without_empty_lists, "Topic": topics })
# documents_in_topic_0 = df[df["Topic"] == 0]["Document"].tolist()
# print(documents_in_topic_0)
# fig = topic_model.visualize_topics()
# fig.write_html('file.html')



heads = topic_model.get_topic_freq()
# trae solo 3 documentos representativos por topicos, estaria bueno que traiga más ~Fede2023

for head in heads['Topic']:
    if(head != -1):
        print(f"Topico {head}: \n{topic_model.get_topic(head)}\n")
        print(f"Documentos representativos: \n{topic_model.get_representative_docs(head)}\n")

#EVALUATION
documents = pd.DataFrame({"Document": processed_data,
                          "ID": range(len(processed_data)),
                          "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

# Extract vectorizer and analyzer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names_out()
tokens = [analyzer(doc) for doc in processed_data]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

print(topic_words, 
                                 tokens, 
                                 corpus,
                                 dictionary);
# Evaluate
coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='c_v')
coherence = coherence_model.get_coherence()

print(coherence)

# Obtener el diccionario de tópicos
# topics_dict = topic_model.get_topic_info()
# print(f"Topicos: \n{topics_dict}\n")

# i = -2
# while(i<10):
#     print(f"Topico {i}: \n{topic_model.get_topic(i)}\n");
#     i = i+1;



# Guardamos en un archivo
# df = pd.DataFrame(topics_dict)
# df.to_csv('topics_dict.csv', index=False)



# Ejecutar con: python C:/Users/gonza/Desktop/J/wFacu/Tesina/BERTopic/BERTopic-1.py