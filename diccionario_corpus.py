import csv
from unidecode import unidecode
# pip install -U pip setuptools wheel
# pip install -U spacy
# python -m spacy download es_core_news_sm

from spellchecker import SpellChecker # lematizacion

# Cargar el diccionario de la RAE en un conjunto para búsquedas rápidas
def cargar_diccionario(ruta_diccionario):
    with open(ruta_diccionario, 'r', encoding='utf-8') as archivo:
        # Usamos un set para una búsqueda rápida, eliminando los saltos de línea
        diccionario = set(palabra.strip() for palabra in archivo)

    return diccionario


def cargar_nombres_en_conjunto(nombre_archivo):
    conjunto_nombres = set()
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        next(lector)  # Saltar la fila de encabezados
        for fila in lector:
            nombre = unidecode(fila[0]).lower()
            conjunto_nombres.add(nombre)
    return conjunto_nombres

# Leer el archivo de narrativa, dividir en palabras, y comparar con el diccionario
def verificar_narrativa(ruta_narrativa, diccionario, diccionario_nombres, ruta_no_encontradas, ruta_palabras_a_corregir):
    # Creamos un conjunto para almacenar las palabras no encontradas y evitar duplicados
    palabras_no_encontradas = set()
    diccionario_no_encontradas = {}
    
    with open(ruta_narrativa, 'r', encoding='utf-8') as archivo:
        texto = archivo.read()

        # Dividir línea en palabras y limpiar caracteres no deseados
        palabras = texto.split()
        for palabra in palabras:
            # Verificar si la palabra está en el diccionario
            if (
                palabra not in diccionario_nombres 
                and palabra not in diccionario
                # and palabra not in palabras_no_encontradas
                and aplicar_correcciones(palabra) not in diccionario
            ):
                if (palabra in diccionario_no_encontradas):
                    diccionario_no_encontradas[palabra] += 1
                else:
                    diccionario_no_encontradas[palabra] = 1

                palabras_no_encontradas.add(palabra)

    # Guardar palabras a corregir en el archivo de salida
    with open(ruta_palabras_a_corregir, 'w', encoding='utf-8') as archivo_salida:
        for palabra in sorted(diccionario_no_encontradas):  # Ordenar alfabéticamente
            archivo_salida.write(palabra + ': ' + str(diccionario_no_encontradas[palabra]) + '\n')

    # Guardar palabras no encontradas en el archivo de salida
    with open(ruta_no_encontradas, 'w', encoding='utf-8') as archivo_salida:
        for palabra in sorted(palabras_no_encontradas):  # Ordenar alfabéticamente
            archivo_salida.write(palabra + '\n')

def aplicar_correcciones(palabra):
    spell = SpellChecker(language='es')

    # find those words that may be misspelled
    misspelled = spell.unknown(palabra)
    # Obtener la corrección sugerida
    correccion = spell.correction(palabra)

    # Si hay una corrección y es diferente a la palabra original, devolverla
    if correccion and correccion != palabra:
        print('word: ' + (palabra or '') + ', correction: ' + (correccion or '') + ', candidates: ' + str(spell.candidates(palabra) or ''))
        return correccion # Ojo con esto, porque la correccion la devuelve con acentos y en el diccionario creo que nosotros los filtramos
    else:
        # Devolver la palabra original si no hay corrección
        return palabra

# Especificar rutas de los archivos
ruta_diccionario_nombres = 'own/dictionary/dictionary_names.csv'
ruta_diccionario = 'own/dictionary/dictionary.txt'
ruta_narrativa = 'own/datasets/tweets_municipalidad_short.txt'
ruta_no_encontradas = 'own/preprocess/palabras_no_encontradas.txt'
ruta_palabras_a_corregir = 'own/preprocess/diccionario_no_encontradas.txt'

# Cargar el diccionario y ejecutar la verificación
diccionario = cargar_diccionario(ruta_diccionario)
diccionario_nombres = cargar_nombres_en_conjunto(ruta_diccionario_nombres)
verificar_narrativa(ruta_narrativa, diccionario, diccionario_nombres, ruta_no_encontradas, ruta_palabras_a_corregir)

print("Proceso completado. Las palabras no encontradas se han guardado en:", ruta_no_encontradas)