from octis.dataset.dataset import Dataset
from octis.dataset.dataset import Dataset
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.models.LDA import LDA

# Define the search space. To see which hyperparameters to optimize, see the topic model's initialization signature
search_space = {"alpha": Real(low=0.001, high=5.0), "eta": Real(low=0.001, high=5.0)}

dataset_path = "/app/own/datasets/tweets_municipalidad"
dataset = Dataset()
dataset.load_custom_dataset_from_folder(dataset_path)

npmi = Coherence(texts=dataset.get_corpus())

# Initialize an optimizer object and start the optimization.
optimizer=Optimizer()
opt_result=optimizer.optimize(model=LDA(),
                            dataset=dataset,
                            metric=npmi,
                            search_space=search_space,
                            save_models=True,
                            save_path="./results/cross_validation",
                            number_of_call=30, # number of optimization iterations
                            model_runs=5) # number of runs of the topic model

#save the results of th optimization in a csv file
opt_result.save_to_csv("results.csv")