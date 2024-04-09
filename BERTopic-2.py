import sys
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
# from sklearn.datasets import fetch_20newsgroups

# Importaciones para el preprocesamiento
from nltk.corpus import stopwords
import spacy
from spacy.lang.es import Spanish
import nltk
# nltk.download('stopwords')
from unidecode import unidecode

# Nuestros Tuits
docs = pd.read_csv('C:/Users/gonza/Desktop/J/wFacu/Tesina/BERTopic/tweets_municipalidad-short.csv')
docs = docs['tweet'].astype(str)

# Preprocesamiento
nlp = spacy.load('es_core_news_sm') # Spanish
spanish_stopwords = set(stopwords.words('spanish'))

def preprocess(text):
    result = ''
    doc = nlp(text)
    for token in doc:
        if token.is_alpha and token.text.lower() not in spanish_stopwords and len(token.text) > 3:
            token_text = unidecode(token.text.lower())  # Aplicar unidecode al texto del token
            result = result + ' ' + token_text if result else token_text
    return result


processed_data = docs.map(preprocess)

# Quitamos arrays vacios
processed_data_without_empty_lists = [item for item in processed_data if item]

# print(processed_data_without_empty_lists)
# sys.exit()

# Inicializamos el modelo


# Fine-tune your topic representations
representation_model = KeyBERTInspired()
topic_model = BERTopic(language="spanish", representation_model=representation_model)

# Corremos el modelo
topics, probs = topic_model.fit_transform(processed_data_without_empty_lists)
fig = topic_model.visualize_topics()
fig.write_html('file.html')


# TODO buscar un lemmatizador o similar
# TODO podemos obtener oraciones de referencia de cada topico


heads = topic_model.get_topic_freq().head()

for head in heads['Topic']:
    if(head != -1):
        print(f"Topico {head}: \n{topic_model.get_topic(head)}\n");

# Obtener el diccionario de tópicos
topics_dict = topic_model.get_topic_info()
print(f"Topicos: \n{topics_dict}\n")

i = -2
while(i<10):
    print(f"Topico {i}: \n{topic_model.get_topic(i)}\n");
    i = i+1;



# Guardamos en un archivo
df = pd.DataFrame(topics_dict)
df.to_csv('topics_dict.csv', index=False)



# Ejecutar con: python C:/Users/gonza/Desktop/J/wFacu/Tesina/BERTopic/BERTopic-1.py