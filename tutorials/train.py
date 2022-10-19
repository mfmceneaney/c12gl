#----------------------------------------------------------------------#
# Example dataset creation script
# Author: Matthew McEneaney
# Contact: matthew.mceneaney@duke.edu
#----------------------------------------------------------------------#

# PyTorch Imports
import torch.nn as nn
import ignite.distributed as idist

# Custom Imports
from c12gl.models import GIN, MLP
from c12gl.dataloading import getGraphDatasetInfo, loadGraphDataset
from c12gl.utils import setPltParams, train, trainDA, evaluate, evaluateOnData

# Random Imports
import numpy as np
import matplotlib.pyplot as plt
import os

#-------------------- TRAINING --------------------#
#import sys
#sys.path.append('/Users/mfm45/c12gl')
#from c12gl.dataloading import loadGraphDataset

#TODO: LOADGRAPHDATASET METHOD IN C12GL
#TODO: CHECK LABELLING IN DATASET CREATION...
#TODO: TRY ADDING EVNUM AS TEST DATASET LABEL AND OTHER STUFF.... WITH PARENT INDEXING....
#TODO: UPDATE C12ANALYSIS TO READ IN CSV FILE TO HASHMAP  ----> SAVED PAGE IN SAFARI

dataset = 'test_dataset_10_18_22'
split = [0.8,0.1,0.1] #NOTE: #TODO: CONVERT THIS TO FRACTIONS AND UPDATE LOAD_DATASET METHOD IN C12GL
max_events = 2000
batch_size = 32
num_workers = 0
loaders = loadGraphDataset(
                            dataset=dataset,
                            prefix='',
                            key='data',
                            ekey='',
                            split=split,
                            max_events=max_events,
                            batch_size=batch_size,
                            drop_last=False,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            verbose=True
                            )
train_loader, val_loader, test_loader, nclasses, ndata_dim, edata_dim = loaders

#-------------------- TRAINING --------------------#
import torch
import torch.optim as optim
device = torch.device('cpu')
num_layers = 5
num_mlp_layers = 3
input_dim = ndata_dim
hidden_dim = 32
output_dim = nclasses
final_dropout = 0.5
learn_eps = False
graph_pooling_type = 'mean'
neighbor_pooling_type = 'mean'
model = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type).to(device)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
gamma = 0.1
patience = 10
threshold = 0.001
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                                optimizer,
                                                mode='min',
                                                factor=gamma,
                                                patience=patience,
                                                threshold=threshold,
                                                threshold_mode='rel',
                                                cooldown=0,
                                                min_lr=0,
                                                eps=1e-08,
                                                verbose=True
                                                )
criterion = nn.CrossEntropyLoss() #BITEL
max_epochs = 10
dataset = "test_dataset"
prefix = ""
log_interval = 10
log_dir = "logs" #NOTE: #TODO: CHECK osp.expanduser and full path in c12gl....
model_name = 'model'
verbose = True

rank = 0
config = {
    'model'        : model,
    'device'       : torch.device('cpu'),
    'train_loader' : train_loader,
    'val_loader'   : val_loader,
    'optimizer'    : optimizer,
    'scheduler'    : scheduler,
    'criterion'    : criterion,
    'max_epochs'   : max_epochs,
    'dataset'      : dataset,
    'prefix'       : prefix,
    'log_interval' : log_interval,
    'log_dir'      : log_dir,
    'model_name'   : model_name,
    'verbose'      : verbose
}
import mlflow

# Create mlflow experiment and set...
try: 
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(os.path.join(log_dir,"mlruns"),exist_ok=True)
except FileExistsError: print("Directory:",log_dir,"/mlruns already exists!")

"""
tracking_uri= os.path.join(log_dir,"mlruns") #OLD: log_dir
print("DEBUGGING: tracking_uri = ",tracking_uri)#DEBUGGING
mlflow.set_tracking_uri(tracking_uri)#DEBUGGING: ADDED

# Create experiment
experiment_name = "test_experiment"#DEBUGGING
#mlflow.delete_experiment(experiment_name)
experiment_id = mlflow.create_experiment(
    experiment_name,
    artifact_location=os.path.join(log_dir,"mlruns"),
    tags={"version": "v1", "priority":"P1"}
)

# Get experiment info
experiment = mlflow.get_experiment(experiment_id)
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# Set experiment
experiment = mlflow.set_experiment(experiment.name)
print("DEBUGGING: experiment = ",experiment)#DEBUGGING
"""

# tracking_uri = Path.absolute(Path(args.log).joinpath("mlruns")).as_uri()

# Train model
train(rank,config)

"""
# Try training in distributed configuration
config['distributed'] = True

backend = "nccl"  # torch native distributed configuration on multiple GPUs
# backend = "xla-tpu"  # XLA TPUs distributed configuration
# backend = None  # no distributed configuration
#
dist_configs = {'nproc_per_node': 2}  # Use specified distributed configuration if launch as python main.py
dist_configs["start_method"] = "spawn" ##"fork"  # Add start_method as "fork" if using Jupyter Notebook

config = locals()#NOTE: ADDED
print("DEBUGGING: config = ",config)#DEBUGGING
with idist.Parallel(backend=backend, **dist_configs) as parallel:
    parallel.run(train, config)
"""

eval_results = evaluate(
    model,
    device,
    test_loader=test_loader,
    dataset='',
    prefix='',
    max_events=0,
    verbose=verbose
    )

test_acc, probs_Y, argmax_Y, decisions_true, decisions_false = eval_results

print("DEBUGGING: np.shape(argmax_Y)        = ",np.shape(argmax_Y))#DEBUGGING
print("DEBUGGING: np.shpae(decisions_true)  = ",np.shape(decisions_true))#DEBUGGING
print("DEBUGGING: np.shpae(decisions_false) = ",np.shape(decisions_false))#DEBUGGING

print("DONE")
