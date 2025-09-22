# DOCKER

## Instalación
1. build: `docker build -t bertopic .`, agregando `--platform linux/amd64` para Apple Silicon.
2. para que se repliquen los cambios: `docker run -v /Users/federuhl/Documents/pruebas-tesina/BERTopic:/app bertopic`

## Versiones compatibles
El proyecto usa versiones específicas de dependencias para garantizar compatibilidad:
- **Python**: 3.9
- **numpy**: 1.26.4
- **scikit-learn**: 1.6.1
- **pandas**: 2.3.2
- **bertopic**: 0.9.4
- **octis**: 1.14.0
- **umap-learn**: 0.5.9.post2

Todas las versiones exactas están documentadas en `requirements.txt`.

# BERTopic
### Para correr
1. cd own/models/BERTopic
2. python validate_models.py

## Para las validaciones 
### Resultados de los autores de BERTopic
- 20 NewsGroups: 
    - **TC = .166**
    - **TD = .851**
- BBC News: 
    - **TC = .167**
    - **TD = .794**
- Trump: 
    - **TC = .066**
    - **TD = .663**

## Resultados tweets_municipalidad
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.47897  |
| **npmi**      | 0.058348 |
| **diversity** | 0.800105 |

## Resultados senadores
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.535749 |
| **npmi**      | 0.193836 |
| **diversity** | 0.637442 |

## Resultados elecciones Argentina
| Métrica       | Valor     |
|---------------|-----------|
| **cv**        | 0.361299  |
| **npmi**      | -0.010353 |
| **diversity** | 0.805048  |

# LDA
### Para correr
1. cd own/models/LDA
2. python validate_models.py

## Para las validaciones 
### Resultados de los autores de BERTopic
- 20 NewsGroups: 
    - **TC = .058**
    - **TD = .749**
- BBC News: 
    - **TC = .014**
    - **TD = .577**
- Trump: 
    - **TC = -.011**
    - **TD = .502**

## Resultados tweets_municipalidad
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.364842 |
| **npmi**      |-0.029664 |
| **diversity** | 0.897936 |

##  Resultados de tweets_municipalidad tras aplicar el optimizador de octis (npmi) para alpha y eta
### "alpha": 4.38393917610078, "eta": 4.9725466249069346
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.360588 |
| **npmi**      |-0.011460 |
| **diversity** | 0.497913 |

## Resultados senadores
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.398367 |
| **npmi**      | 0.014036 |
| **diversity** | 0.247566 |

## TODO
- [x] validar con [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/pdf/2203.05794).
    >[!NOTE]
    *Ranging from 10 to 50 topics with steps of 10, topic coherence (TC) and topic diversity (TD) were calculated at each step for each topic model. All results were averaged across 3 runs for each step. Thus, each score is the average of 15 separate runs* (https://arxiv.org/pdf/2203.05794).

# NMF
### Para correr
1. cd own/models/NMF
2. python validate_models.py

## Para las validaciones 
### Resultados de los autores de BERTopic
- 20 NewsGroups: 
    - **TC = .089**
    - **TD = .663**
- BBC News: 
    - **TC = .012**
    - **TD = .549**
- Trump: 
    - **TC = .009**
    - **TD = .379**

## Resultados tweets_municipalidad
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.454306 |
| **npmi**      | 0.051452 |
| **diversity** | 0.595551 |

## Resultados senadores
| Métrica       | Valor    |
|---------------|----------|
| **cv**        | 0.487169 |
| **npmi**      | 0.032270 |
| **diversity** | 0.492323 |

# General
> [!IMPORTANT]
> El autor, para medir la coherencia, utilizo `npmi`
> Los archivos encontrados en `evaluation/results.py` y `evaluation/evaluation.py` del autor, fueron modificados para trabajar con metrica c_v. Por lo que, si los archivos JSONs no tienen dicha metrica, `results` devolvera "NaN" en la columna correspondiente.
> Funciones como las graficas de tablas, encontradas en `evaluation/results.py` (ej: visualize_table_tq), no fueron probadas, siendo probable que fallen por el cambio a las metricas.
> Para validar el modelo, el número de topicos se iteró como dice el autor (`"num_topics": [(i+1)*10 for i in range(5)]`). Sin embargo, por fines practicos, se aumentó el rango de topicos (`"num_topics": [(i+1)*5 for i in range(10)]`). Es decir, de la primera forma el numero de topicos es de: [10, 20, 30, 40, 50] y de la segunda: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

## TODO
- [x] Correr todos los modelos con los dataset que usaron los autores.
- [x] Validar todos los modelos.
- [x] Correr todos los modelos con el dataset "tweets_municipalidad".
- [x] Correr todos los modelos con el dataset "senadores".
- [ ] Corregir funciones de graficas (ver notas importantes ^).
- [ ] Intentar mejorar los parametros con cv.

> [!NOTE]
> BERTopic_tweets_municipalidad preprocesada da mejores resultados en npmi y diversity que la version short, tambien preprocesada
> BERTopic_tweets_municipalidad preprocesada da resultados muy similares respecto a npmi, pero mejora la diversity en relacion al BERTopic_tweets_municipalidad sin preprocesar

> [!WARNING]
> Los resultados de show_results (BERTopic) son promedio de promedios, es decir, por cada JSON se calcula el promedio de las metricas y estos promedios se promedian entre si.

# Notas de autores
> [!NOTE]
> The code and results for the experiments in [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](http://arxiv.org/abs/2203.05794).<br>
> The results for Table 1 and 2 can be found in `results/Basic/`. The results for Table 3 can be found in `results/Dynamic Topic Modeling`.<br>
> To run the experiments, you can follow along with the tutorial in `notebooks/Evaluation.ipynb`. <br>
> To visualize the results from the paper, use `notebooks/Results.ipynb`.<br>