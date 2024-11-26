#=========================================
#
# tweets_municipalidad CSV to TXT and preprocessing || En el preprocesamiento spanish_nlp ocupa otras versiones de los packages usados en las validaciones
#
#=========================================

import csv, re
from spanish_nlp import preprocess
import nltk
import unicodedata
import stanza
import os
import zipfile

nltk.data.path.append(r'C:\Users\gonza\AppData\Roaming\nltk_data') # para buscar los recursos de NLTK

input_file = 'own/datasets/tweets_municipalidad_short.csv'
output_file = 'own/datasets/tweets_municipalidad_short.txt'

sp = preprocess.SpanishPreprocess(
    lower=True,
    remove_url=True,
    remove_hashtags=False,
    split_hashtags=True,
    normalize_breaklines=True,
    remove_emoticons=True,
    remove_emojis=True,
    convert_emoticons=False,
    convert_emojis=False,
    normalize_inclusive_language=True,
    reduce_spam=True,
    remove_vowels_accents=False,
    remove_multiple_spaces=True,
    remove_punctuation=True,
    remove_unprintable=True,
    remove_numbers=True,
    remove_stopwords=True,
    stopwords_list='spacy', #'default', 'extended', 'nltk', 'spacy' #justificar porqué spacy habria que probar todos
    lemmatize=True,
    stem=False,
    remove_html_tags=True,
)

def normalize_word(word):
    # Filtra solo letras
    word = ''.join([char for char in word if char.isalpha()])

    return word

def remove_twitter_pics(text):
    # Definir el patrón para las URLs de pic.twitter.com/**
    pattern = r'pic\.twitter\.com/\w+'
    
    # Reemplazar las coincidencias del patrón con una cadena vacía
    clear_text = re.sub(pattern, '', text)
    
    # Retornar el texto limpio
    return clear_text

def normalize_laugh(text):
    # Definir un patrón más flexible para risas con al menos tres caracteres
    pattern = [
        r'\b(?:[jhaeiou]{3,}r?)\b',  # Captura combinaciones de "j", "h", "a", "e", "i", "o", "u" repetidas al menos tres veces
        r'\blol\b',                # "lol"
    ]

    # Unir los patrones en una expresión regular
    regular_expression = '|'.join(pattern)

    # Aplicar la normalización
    normalized_text = re.sub(regular_expression, 'jaja', text, flags=re.IGNORECASE)

    return normalized_text

def configurar_modelo_local(idioma="es", model_dir="stanza_resources"):
    """
    Configura Stanza para usar modelos descargados manualmente.
    :param idioma: Código del idioma, por ejemplo, 'es' para español.
    :param model_dir: Directorio donde se encuentran los modelos descargados.
    :return: None.
    """
    # Ruta esperada para el modelo del idioma
    modelo_idioma = os.path.join(model_dir, idioma, "default.zip")

    # Verificar si el archivo existe
    if not os.path.exists(modelo_idioma):
        raise FileNotFoundError(f"No se encontró el modelo '{modelo_idioma}'. Verifica la ubicación.")

    # Crear el directorio de destino si no existe
    modelo_destino = os.path.join(model_dir, idioma)
    os.makedirs(modelo_destino, exist_ok=True)

    # Extraer el modelo si no está descomprimido
    if not os.path.exists(os.path.join(modelo_destino, "default")):
        with zipfile.ZipFile(modelo_idioma, "r") as zip_ref:
            zip_ref.extractall(modelo_destino)
        print(f"Modelo descomprimido en: {modelo_destino}")

def inicializar_stanza(idioma="es", model_dir="./stanza_resources"):
    """
    Inicializa el pipeline de Stanza utilizando un modelo local.
    :param idioma: Código del idioma, por ejemplo, 'es' para español.
    :param model_dir: Directorio donde se encuentran los modelos.
    :return: Pipeline de Stanza.
    """
    # Configurar modelos locales
    configurar_modelo_local(idioma, model_dir)

    # Inicializar el pipeline
    return stanza.Pipeline(lang=idioma, processors="tokenize,mwt,pos,lemma", model_dir=model_dir)

def lematizar_palabras(palabras, nlp):
    """
    Lematiza una lista de palabras utilizando el pipeline de Stanza.
    :param palabras: Lista de palabras a lematizar.
    :param nlp: Pipeline de Stanza.
    :return: Diccionario con palabras originales y sus lemas.
    """
    texto = " ".join(palabras)  # Combinar palabras en un solo texto
    doc = nlp(texto)            # Procesar el texto con Stanza
    lemas = {}

    # Extraer las palabras y sus lemas, omitiendo las que no cambian
    for sent in doc.sentences:
        for word in sent.words:
            lemas[word.text] = word.lemma

    return lemas

def procesar_palabras(palabras, idioma="es", nlp):
    """
    Procesa un archivo de palabras no encontradas y guarda las lematizaciones en otro archivo.
    :param idioma: Código del idioma, por ejemplo, 'es' para español.
    """
    
    # Eliminar duplicados para optimizar el procesamiento
    palabras_unicas = list(set(palabras))
    
    # Inicializar Stanza con el modelo local
    nlp = inicializar_stanza(idioma=idioma)
    
    # Lematizar palabras
    lemas = lematizar_palabras(palabras_unicas, nlp)
    print(lemas)

    return lemas

# Abrimos el archivo CSV y crea el TXT
with open(input_file, mode='r', encoding='utf-8') as csv_file, open(output_file, mode='w', encoding='utf-8') as txt_file:
    csv_reader = csv.reader(csv_file)
    
    # Saltamos la cabecera
    next(csv_reader)
    
    for row in csv_reader:
        tweet = row[1]

        # Borramos los enlaces a las imagenes
        preprocessed_tweet = remove_twitter_pics(tweet)

        # Eliminamos los saltos de linea
        preprocessed_tweet = sp.transform(preprocessed_tweet, debug=False).strip()

        # Remplazamos los jaja's
        preprocessed_tweet = normalize_laugh(preprocessed_tweet)

        # Lematizamos las palabras
        palabras = preprocessed_tweet.split()  # Dividir el tweet en palabras
        lemas = procesar_palabras(palabras)  # Obtener las palabras lematizadas

        # Unimos las palabras lematizadas en un string
        preprocessed_tweet = " ".join(lemas)

        # Eliminamos caracteres indeseados
        aux = ''
        for word in preprocessed_tweet.split():
            normalized_word = normalize_word(word)
            if len(normalized_word) > 2:
                aux += ' ' + normalized_word
        preprocessed_tweet = aux

        if preprocessed_tweet:
            print(preprocessed_tweet)
            txt_file.write(preprocessed_tweet + '\n')