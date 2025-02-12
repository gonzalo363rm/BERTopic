The code and results for the experiments in [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](http://arxiv.org/abs/2203.05794). 
The results for Table 1 and 2 can be found in `results/Basic/`. The results for Table 3 can be found in `results/Dynamic Topic Modeling`. 

To run the experiments, you can follow along with the tutorial in `notebooks/Evaluation.ipynb`. 
To visualize the results from the paper, use `notebooks/Results.ipynb`. 

<!-- NOTAS -->
BERTopic_tweets_municipalidad preprocesada da mejores resultados en npmi y diversity que la version short, tambien preprocesada
BERTopic_tweets_municipalidad preprocesada da resultados muy similares respecto a npmi, pero mejora la diversity en relacion al BERTopic_tweets_municipalidad sin preprocesar

<!-- TODO -->
*Justificar porque usamos spanish_nlp*

<!-- DOCKER -->
<!-- Build -->
docker build -t bertopic .
<!-- Run and replicate changes -->
docker run -v C:\Users\gonza\Desktop\J\wFacu\Tesina\Modelos\BERTopic:/app bertopic
<!-- TODO: spanish_nlp (que usamos solo para el preprocesamiento) actualiza las versiones de algunos de los package que usamos, deberiamos o usar otra version que no lo haga o quizas mas facil, hacer el preprocesamiento aparte -->
<!-- TODO: Ver que la carpeta trump y trump.txt se generen dentro de datasets -->



Para el filtro de palabras, podriamos considerar que si una palabra de "palabras_no_encontradas" aparece más de n veces, esta bien escrita pero no esta en el diccionario, entonces deberiamos agregarla al mismo
Ver de usar este package: https://pypi.org/project/pyspellchecker/

# LDA

## Para correr
    > cd own/LDA
    > python validate_models.py

## Para las validaciones 
1. Resultados de los autores de BERTopic
    - 20 NewsGroups: 
        - **TC = .058**
        - **TD = .749**
    - BBC News: 
        - **TC = .014**
        - **TD = .577**
    - Trump: 
        - **TC = -.011**
        - **TD = .502**
>[!NOTE] Ranging from 10 to 50 topics with steps of 10, topic coherence (TC) and topic diversity (TD) were calculated at each step for each topic model. All results were averaged across 3 runs for each step. Thus, each score is the average of 15 separate runs.

## TODO
- [x] validar con [https://arxiv.org/pdf/2203.05794]