from evaluation import Results

print("##########\nRESULTADOS DEL AUTOR\n##########")
results = Results("../results/", combine_models=True); results.get_keys()

print("##########\nDATASET BBC NEWS\n##########")
print(results.get_data("BBC_news", aggregated=True))

print("##########\nDATASET 20 NEWS GROUP\n##########")
print(results.get_data("20NewsGroup", aggregated=True))

print("##########\nDATASET TRUMP\n##########")
print(results.get_data("trump", aggregated=True))

##########

print("##########\nRESULTADOS PROPIOS\n##########")
results2 = Results("/app/own/results/", combine_models=True); results2.get_keys()

print("##########\nDATASET BBC NEWS\n##########")
print(results2.get_data("BBC_News", aggregated=True))

print("##########\nDATASET 20 NEWS GROUP\n##########")
print(results2.get_data("20NewsGroup", aggregated=True))

print("##########\nDATASET TRUMP\n##########")
print(results2.get_data("trump", aggregated=True))

print("##########\nDATASET TWEETS MUNICIPALIDAD\n##########")
print(results2.get_data("/app/own/datasets/tweets_municipalidad", aggregated=True))

print("##########\nDATASET TWEETS MUNICIPALIDAD SHORT\n##########")
print(results2.get_data("/app/own/datasets/tweets_municipalidad_short", aggregated=True))