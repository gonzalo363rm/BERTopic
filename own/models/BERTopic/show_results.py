from evaluation import Results

# El autor no calcula la coherence c_v y nosotro modificamos results.py, entonces tira error
print("##########\nRESULTADOS DEL AUTOR\n##########")
results = Results("../../../results/", combine_models=True);

# Ver qué datasets están disponibles
print("Datasets disponibles:", results.get_keys())

print("##########\nDATASET BBC NEWS\n##########")
print(results.get_data("BBC News", aggregated=True))

print("##########\nDATASET 20 NEWS GROUP\n##########")
print(results.get_data("20NewsGroup", aggregated=True))

print("##########\nDATASET TRUMP\n##########")
print(results.get_data("Trump", aggregated=True))

##########

print("##########\nRESULTADOS PROPIOS\n##########")
results2 = Results("/app/own/models/BERTopic/results/", combine_models=True);

# Ver qué datasets están disponibles
print("Datasets disponibles:", results2.get_keys())

print("##########\nDATASET BBC NEWS\n##########")
print(results2.get_data("BBC_News", aggregated=True))

print("##########\nDATASET 20 NEWS GROUP\n##########")
print(results2.get_data("20NewsGroup", aggregated=True))

print("##########\nDATASET TRUMP\n##########")
print(results2.get_data("Trump", aggregated=True))

# print("##########\nDATASET TWEETS MUNICIPALIDAD\n##########")
# print(results2.get_data("tweets_municipalidad", aggregated=True))

# print("##########\nDATASET SENADORES\n##########")
# print(results2.get_data("senadores", aggregated=True))

print("##########\nDATASET TWEETS PREPROCESSED\n##########")
print(results2.get_data("tweets_preprocessed", aggregated=True))

# print("##########\nDATASET TWEETS BALOTAJE\n##########")
# print(results2.get_data("tweets_balotaje_preprocessed", aggregated=True))