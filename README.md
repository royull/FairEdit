# EECS545

# Requirements

Use the environment.yml to find the associated packages and versions. 

# Datasets 

Example data loaders can be found in data_loader_example.py 

Can be run by passing in an argument denoting the data set to load, such as data_loader_example.py -dataset credit

# Models

Folder models holds the various architectures. Examples to load, train, and evaluate can be found in run_models.py. Run_models.py has a lot of possible arguments, here as an example to run:

python run_models.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed 1

where --model can be ['gcn', 'sage'] and --dataset can be ['german', 'credit', 'bail'] 

model weights are saved to weights folder and evaluation metricss are saved to results. 

# comparison_training.py

contains models to compare against. Example:

python comparison_training.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model fairgnn --dataset german --seed 1 --num_layers 32

where --model can be ['fairgnn', 'fairwalk'] and --dataset can be ['german', 'credit', 'bail'] 


# To perform various types of training, code can be found in the training_methods folder

In here, you will find methods such as nifty, brute_force, and fairedit, which will all incorporate some form or fair training.  
