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