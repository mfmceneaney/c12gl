#--------------------------------------------------#
# Description: Utility functions and classes for 
#   training and evaluating DGL GNNs.
# Author: Matthew McEneaney
#--------------------------------------------------#

# ML Imports
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# DGL Graph Learning Imports
import dgl #NOTE: for dgl.batch and dgl.unbatch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info, Subset

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch Ignite Imports
import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import global_step_from_engine, EarlyStopping
from ignite.contrib.handlers.mlflow_logger import MLflowLogger

# Utility Imports
import datetime, os, itertools

# Local Imports
from .dataloading import getGraphDatasetInfo, loadGraphDataset, GraphDataset

#TODO: Fix/update definitions

#------------------------- Functions -------------------------#
# setPltParams
# train
# trainDA
# evaluate
# evaluateOnData

def setPltParams(
    fontsize=20,
    axestitlesize=25,
    axeslabelsize=25,
    xticklabelsize=25,
    yticklabelsize=25,
    legendfontsize=25
    ):
    """
    Arguments
    ---------
    fontsize : int, default 20
    axestitlesize : int, default 25
    axeslabelsize : int, default 25
    xticklabelsize : int, default 25
    yticklabelsize : int, default 25
    legendfontsize : int, default 25

    Description
    -----------
    Set font sizes for matplotlib.plt plots.
    """
    plt.rc('font', size=20) #controls default text size                                                                                                                     
    plt.rc('axes', titlesize=25) #fontsize of the title                                                                                                                     
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels                                                                                                            
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels                                                                                                            
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels                                                                                                            
    plt.rc('legend', fontsize=15) #fontsize of the legend

def train(
    rank,
    config
    ):

    """
    Parameters
    ----------
    rank : int
    config : dict

    Entries in config
    -----------------
    model : torch.nn.Module, required
    device : str, required
    train_loader : dgl.dataloading.GraphDataloader, required
    val_loader : dgl.dataloading.GraphDataloader, required
    optimizer : torch.optim.optimizer, required
    scheduler : torch.optim.lr_scheduler, required
    criterion : torch.nn.loss, required
    max_epochs : int, required
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    log_interval : int, optional
        Default : 10
    log_dir : str, optional
        Default : "logs/"
    model_name : str, optional
        Default : "model"
    verbose : bool, optional
        Default : True
    distributed : boolean, optional
        Default : False
    mlflow : boolean, optional
        Default : False

    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch

    Description
    -----------
    Train a GNN using a basic supervised learning approach.
    """

    # Get configuration parameters
    model        = config['model']
    device       = config['device']
    train_loader = config['train_loader']
    val_loader   = config['val_loader']
    optimizer    = config['optimizer']
    scheduler    = config['scheduler']
    criterion    = config['criterion']
    max_epochs   = config['max_epochs']
    log_interval = config['log_interval']
    log_dir      = config['log_dir']
    model_name   = config['model_name']
    verbose      = config['verbose']
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create logs for metrics
    logs={'train':{'loss':[],'accuracy':[]}, 'val':{'loss':[],'accuracy':[]}}

    # Distributed setup
    if 'distributed' in config.keys() and config['distributed']:
        
        # Show info if requested
        if verbose: print(
            idist.get_rank(),
            ": run with config:",
            config,
            "- backend=",
            idist.backend(),
            "- world size",
            idist.get_world_size(),
        )

        device = idist.device()

        # Convert dataloaders to distributed form
        train_loader = idist.auto_dataloader(
            train_loader.dataset,
            collate_fn=train_loader.collate_fn,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            shuffle=True,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last
        )
        val_loader = idist.auto_dataloader(
            val_loader.dataset,
            collate_fn=val_loader.collate_fn,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            shuffle=True,
            pin_memory=val_loader.pin_memory,
            drop_last=val_loader.drop_last
        )

        # Model, criterion, optimizer setup
        model = idist.auto_model(model)
        optimizer = idist.auto_optim(optimizer)

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

        # Get predictions and loss from data and labels
        x    = batch[0].to(device) # Batch data
        y    = batch[1][:,0].clone().detach().long().to(device) if len(np.shape(batch[1]))==2 else batch[1].clone().detach().long().to(device) #NOTE: This assumes labels is 2D and classification labels are integers
        prediction_raw = model(x) # Model prediction
        loss = criterion(prediction_raw, y) #NOTE: DO NOT APPLY SOFTMAX BEFORE CrossEntropyLoss

        # Step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply softmax and get accuracy
        prediction_softmax = torch.softmax(prediction_raw, 1)
        prediction         = torch.max(prediction_softmax, 1)[1].view(-1, 1)
        acc                = (y.float().view(-1,1) == prediction.float()).sum().item() / len(y)

        return {
                'y': y,
                'prediction_raw': prediction_raw,
                'prediction': prediction,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.

            # Get predictions and loss from data and labels
            x    = batch[0].to(device)
            y    = batch[1][:,0].clone().detach().long().to(device) if len(np.shape(batch[1]))==2 else batch[1].clone().detach().long().to(device) #NOTE: This assumes labels is 2D and classification labels are integers
            h    = model(x)
            prediction_raw = model(x) # Model prediction
            loss = criterion(prediction_raw, y)

        # Apply softmax and get accuracy
        prediction_softmax = torch.softmax(prediction_raw, 1)
        prediction         = torch.max(prediction_softmax, 1)[1].view(-1, 1)
        acc                = (y.float().view(-1,1) == prediction.float()).sum().item() / len(y)

        return {
                'y': y,
                'prediction_raw': prediction_raw,
                'prediction': prediction,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics
    train_accuracy = Accuracy(output_transform=lambda x: [x['prediction'], x['y']])
    train_accuracy.attach(trainer, 'accuracy')
    train_loss     = Loss(criterion,output_transform=lambda x: [x['prediction_raw'], x['y']])
    train_loss.attach(trainer, 'loss')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add evaluation metrics
    val_accuracy = Accuracy(output_transform=lambda x: [x['prediction'], x['y']])
    val_accuracy.attach(evaluator, 'accuracy')
    val_loss     = Loss(criterion,output_transform=lambda x: [x['prediction_raw'], x['y']])
    val_loss.attach(evaluator, 'loss')

#     # Set up early stopping
#     # score_function = config['score_function'] if 'score_function' in config.keys() else score_function
#     def score_function(engine):
#         val_loss = engine.state.metrics['loss']
#         return -val_loss

#     handler = EarlyStopping(
#         patience=patience,
#         min_delta=args.min_delta,
#         cumulative_delta=args.cumulative_delta,
#         score_function=score_function,
#         trainer=trainer
#         )
#     evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

#     # Step learning rate #NOTE: DEBUGGING: TODO: Replace above...
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def stepLR(trainer):
#         if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
#             scheduler.step(trainer.state.output['loss'])#TODO: NOTE: DEBUGGING.... Fix this...
#         else:
#             scheduler.step()
            
    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

    # Log training metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(trainer):
        metrics = evaluator.run(train_loader).metrics
        for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
        if verbose: print(f"\nTraining Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    # Log validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer):
        metrics = evaluator.run(val_loader).metrics
        for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
        if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
        
    # Save model
    torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,model_name)) #NOTE: Save to cpu state so you can test more easily.
   
    # Create training/validation loss plot #NOTE: #TODO: ARE THESE GRAPHS REALLY NECESSARY???
    f_loss = plt.figure()
    plt.subplot()
    plt.title('Loss per epoch')
    plt.plot(logs['train']['loss'],label="training")
    plt.plot(logs['val']['loss'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    f_loss.savefig(os.path.join(log_dir,'training_loss.png'))

    # Create training/validation accuracy plot #NOTE: #TODO: ARE THESE GRAPHS REALLY NECESSARY???
    f_acc = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f_acc.savefig(os.path.join(log_dir,'training_accuracy.png'))
    
    # Run MLFlow routines if requested
    if 'mlflow' in config.keys() and config['mlflow']:

        # Start MLFlow run
        mlflow.start_run()#NOTE: ADDED 10/17/22
        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))

        # Create MLFlow logger
        tracking_uri = config['tracking_uri'] if 'tracking_uri' in config.keys() else '' #TODO: Make this an option
        mlflow_logger = MLflowLogger(tracking_uri=tracking_uri)

        # Log experiment parameters:                                                                                                                                                                                                            
        mlflow.set_tag("trial_number",config['trial_number'] if 'trial_number' in config.keys() else -1)#DEBUGGING ADDED                                                                                                                                                                                        
        mlflow_logger.log_params({
            "seed": config['seed'] if 'seed' in config.keys() else -1,                                                                                                                                                                                                                    
            "batch_size": batch_size,
            "model": model.__class__.__name__,

            "pytorch version": torch.__version__,
            "ignite version": ignite.__version__,
            "cuda version": torch.version.cuda,
            "device name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(-1),
            "trial number": config['trial_number'] if 'trial_number' in config.keys() else -1
        })

        # Attach the logger to the trainer to log training loss at each iteration
        mlflow_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda output : output['loss']
        )

        # Attach the logger to the evaluator on the training dataset after each epoch
        mlflow_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_step_from_engine(trainer),
        )

        # Attach the logger to the evaluator on the validation dataset after each epoch
        mlflow_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_step_from_engine(trainer)
        )

        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
        mlflow_logger.attach_opt_params_handler(
            trainer,
            event_name=Events.ITERATION_STARTED,
            optimizer=optimizer,
            param_name='lr'  # optional
        )
        
        # Save model in MLFlow format
        mlflow_logger.pytorch.log_model(model,model_name)
        
        mlflow.log_figure(f_loss, 'training_loss.png')
        mlflow.log_figure(f_acc,  'training_accuracy.png')
        
        mlflow.end_run()
        
    return logs

def trainDA(
        rank,
        config
    ):

    """
    Parameters
    ----------
    rank : int
    config : dict

    Entries in config
    -----------------
    args : argparse.Namespace, required
    model : torch.nn.Module, required
    classifier : torch.nn.Module, required
    discriminator : torch.nn.Module, required
    device : str, required
    train_loader : dgl.dataloading.GraphDataLoader, required
    val_loader : dgl.dataloading.GraphDataLoader, required
    dom_train_loader : dgl.dataloading.GraphDataLoader, required
    dom_val_loader : dgl.dataloading.GraphDataLoader, required
    model_optimizer : torch.optim.optimizer, required
    classifier_optimizer : torch.optim.optimizer, required
    discriminator_optimizer : torch.optim.optimizer, required
    scheduler : torch.optim.lr_scheduler, required
    train_criterion : torch.nn.loss, required
    dom_criterion : torch.nn.loss, required
    alpha : float, required
    max_epochs : int, required
    log_interval : int, optional
        Default : 10
    log_dir : str, optional
        Default : 'logs'
    model_name : str, optional
        Default : 'model'
    verbose : bool, optional
        Default : True
    distributed : boolean, optional
        Default : False
    mlflow : boolean, optional
        Default : False
    
    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch

    Description
    -----------
    Train a GNN using a Domain Adversarial approach.
    """

    # Get configuration parameters
    model                   = config['model']
    classifier              = config['classifier']
    discriminator           = config['discriminator']
    device                  = config['device']
    train_loader            = config['train_loader']
    val_loader              = config['val_loader']
    dom_train_loader        = config['dom_train_loader']
    dom_val_loader          = config['dom_val_loader']
    model_optimizer         = config['model_optimizer']
    classifier_optimizer    = config['classifier_optimizer']
    discriminator_optimizer = config['discriminator_optimizer']
    scheduler               = config['scheduler']
    train_criterion         = config['train_criterion']
    dom_criterion           = config['dom_criterion']
    max_epochs              = config['max_epochs']
    log_interval            = config['log_interval']
    log_dir                 = config['log_dir']
    model_name              = config['model_name']
    verbose                 = config['verbose']

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Logs for matplotlib plots
    logs={'train':{'train_loss':[],'train_accuracy':[],'dom_loss':[],'dom_accuracy':[]},
            'val':{'train_loss':[],'train_accuracy':[],'dom_loss':[],'dom_accuracy':[]}}

    # Distributed setup
    if 'distributed' in config.keys() and config['distributed']:
        
        # Show info if requested
        if verbose: print(
            idist.get_rank(),
            ": run with config:",
            config,
            "- backend=",
            idist.backend(),
            "- world size",
            idist.get_world_size(),
        )

        device = idist.device()

        # Convert dataloaders to distributed form
        train_loader = idist.auto_dataloader(
            train_loader.dataset,
            collate_fn=train_loader.collate_fn,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            shuffle=True,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last
        )
        val_loader = idist.auto_dataloader(
            val_loader.dataset,
            collate_fn=val_loader.collate_fn,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            shuffle=True,
            pin_memory=val_loader.pin_memory,
            drop_last=val_loader.drop_last
        )

        # Convert domain dataloaders to distributed form
        dom_train_loader = idist.auto_dataloader(
            dom_train_loader.dataset,
            collate_fn=dom_train_loader.collate_fn,
            batch_size=dom_train_loader.batch_size,
            num_workers=dom_train_loader.num_workers,
            shuffle=True,
            pin_memory=dom_train_loader.pin_memory,
            drop_last=dom_train_loader.drop_last
        )
        dom_val_loader = idist.auto_dataloader(
            dom_val_loader.dataset,
            collate_fn=dom_val_loader.collate_fn,
            batch_size=dom_val_loader.batch_size,
            num_workers=dom_val_loader.num_workers,
            shuffle=True,
            pin_memory=dom_val_loader.pin_memory,
            drop_last=dom_val_loader.drop_last
        )

        # Model, criterion, optimizer setup
        model         = idist.auto_model(model)
        classifier    = idist.auto_model(classifier)
        discriminator = idist.auto_model(discriminator)
        model_optimizer         = idist.auto_optim(model_optimizer)
        classifier_optimizer    = idist.auto_optim(classifier_optimizer)
        discriminator_optimizer = idist.auto_optim(discriminator_optimizer)

    # Continuously sample target domain data for training and validation
    dom_train_set = itertools.cycle(dom_train_loader)
    dom_val_set   = itertools.cycle(dom_val_loader)

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

        # Get domain data
        dom_x = dom_train_set.__next__()[0] #NOTE: This returns [dgl.HeteroGraph,torch.tensor] for graph and labels.
        dom_x = dom_x.to(device)

        # Get predictions and loss from data and labels
        train_x      = batch[0].to(device)
        train_labels = batch[1][:,0].clone().detach().long().to(device) if len(np.shape(batch[1]))==2 else batch[1].clone().detach().long().to(device) #NOTE: This assumes labels is 2D and classification labels are integers

        # Concatenate classification data and domain data
        train_x     = dgl.unbatch(train_x)
        dom_x       = dgl.unbatch(dom_x)
        nLabelled   = len(train_x)
        nUnlabelled = len(dom_x)
        train_x.extend(dom_x)
        train_x     = dgl.batch(train_x) #NOTE: Training and domain data must have the same schema for this to work.

        # Get hidden representation from model on training and domain data
        h = model(train_x)
        
        # Step the domain discriminator on training and domain data
        dom_y      = discriminator(h.detach())
        dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
        dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].
        discriminator.zero_grad()
        dom_loss.backward()
        discriminator_optimizer.step()
        
        # Step the classifier on training data
        train_y    = classifier(h[:nLabelled]) #NOTE: Only train on labelled (i.e., training) data, not domain data.
        dom_y      = discriminator(h)
        train_loss = train_criterion(train_y, train_labels)
        dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using nn.Sigmoid() is important since the predictions need to be in [0,1].

        # Get total loss using lambda coefficient for epoch
        tot_loss = train_loss - alpha * dom_loss
        
        # Zero gradients in all parts of model
        model.zero_grad()
        classifier.zero_grad()
        discriminator.zero_grad()
        
        # Step total loss
        tot_loss.backward()
        
        # Step classifier and model optimizers (backwards)
        classifier_optimizer.step()
        model_optimizer.step()

        # Apply softmax and get accuracy on training data
        train_prediction_softmax = torch.softmax(train_y, 1)
        train_prediction         = torch.max(train_prediction_softmax, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        train_acc                = (train_labels == train_prediction.float()).sum().item() / len(train_labels)

        # Apply softmax and get accuracy on domain data
        dom_prediction_softmax = torch.softmax(dom_y, 1)
        dom_prediction         = torch.max(dom_prediction_softmax, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        dom_acc                = (dom_labels == dom_prediction.float()).sum().item() / len(dom_labels)

        return {
                'y': train_labels,
                'prediction_raw': train_y,
                'prediction': train_prediction,
                'loss': train_loss.detach().item(),
                'accuracy': train_acc,
                'dom_y': dom_labels,
                'dom_prediction_raw': dom_labels,
                'dom_prediction': dom_prediction,
                'dom_loss': dom_loss.detach().item(),
                'dom_accuracy': dom_acc,
                'tot_loss': tot_loss.detach().item()
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.
            
            # Get domain data
            dom_x = dom_val_set.__next__()[0] #NOTE: This returns [dgl.HeteroGraph,torch.tensor] for graph and labels.
            dom_x = dom_x.to(device)

            # Get predictions and loss from data and labels
            train_x      = batch[0].to(device)
            train_labels = batch[1][:,0].clone().detach().long().to(device) if len(np.shape(batch[1]))==2 else batch[1].clone().detach().long().to(device) #NOTE: This assumes labels is 2D and classification labels are integers

            # Concatenate classification data and domain data
            train_x     = dgl.unbatch(train_x)
            dom_x       = dgl.unbatch(dom_x)
            nLabelled   = len(train_x)
            nUnlabelled = len(dom_x)
            train_x.extend(dom_x)
            train_x     = dgl.batch(train_x) #NOTE: Training and domain data must have the same schema for this to work.

            # Get hidden representation from model on training and domain data
            h = model(train_x)
            
            # Step the domain discriminator on training and domain data
            dom_y      = discriminator(h.detach())
            dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
            dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].
            
            # Step the classifier on training data
            train_y    = classifier(h[:nLabelled]) #NOTE: Only train on labelled (i.e., training) data, not domain data.
            dom_y      = discriminator(h)
            train_loss = train_criterion(train_y, train_labels)
            dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using nn.Sigmoid() is important since the predictions need to be in [0,1].

            # Get total loss using lambda coefficient for epoch
            tot_loss = train_loss - alpha * dom_loss

            # Apply softmax and get accuracy on training data
            train_prediction_softmax = torch.softmax(train_y, 1)
            train_prediction         = torch.max(train_prediction_softmax, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            train_acc                = (train_labels == train_prediction.float()).sum().item() / len(train_labels)

            # Apply softmax and get accuracy on domain data
            dom_prediction_softmax = torch.softmax(dom_y, 1)
            dom_prediction         = torch.max(dom_prediction_softmax, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            dom_acc                = (dom_labels == dom_prediction.float()).sum().item() / len(dom_labels)

        return {
                'y': train_labels,
                'prediction_raw': train_y,
                'prediction': train_prediction,
                'loss': train_loss.detach().item(),
                'accuracy': train_acc,
                'dom_y': dom_labels,
                'dom_prediction_raw': dom_labels,
                'dom_prediction': dom_prediction,
                'dom_loss': dom_loss.detach().item(),
                'dom_accuracy': dom_acc,
                'tot_loss': tot_loss.detach().item()
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics for classifier
    train_accuracy  = Accuracy(output_transform=lambda x: [x['prediction'], x['y']])
    train_accuracy.attach(trainer, 'train_accuracy')
    train_loss      = Loss(train_criterion,output_transform=lambda x: [x['prediction_raw'], x['y']])
    train_loss.attach(trainer, 'train_loss')

    # Add training metrics for discriminator
    dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_prediction'], x['dom_y']])
    dom_accuracy.attach(trainer, 'dom_accuracy')
    dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_prediction_raw'], x['dom_y']])
    dom_loss.attach(trainer, 'dom_loss')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add validation metrics for classifier
    val_accuracy  = Accuracy(output_transform=lambda x: [x['prediction'], x['y']])
    val_accuracy.attach(evaluator, 'train_accuracy')
    val_loss      = Loss(train_criterion,output_transform=lambda x: [x['prediction_raw'], x['y']])
    val_loss.attach(evaluator, 'train_loss')

    # Add validation metrics for discriminator
    val_dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_prediction'], x['dom_y']])
    val_dom_accuracy.attach(evaluator, 'dom_accuracy')
    val_dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_prediction_raw'], x['dom_y']])
    val_dom_loss.attach(evaluator, 'dom_loss')

    # # Set up early stopping
    # def score_function(engine):
    #     val_loss = engine.state.metrics['train_loss']
    #     return -val_loss

    # handler = EarlyStopping(
    #     patience=args.patience,
    #     min_delta=args.min_delta,
    #     cumulative_delta=args.cumulative_delta,
    #     score_function=score_function,
    #     trainer=trainer
    #     )
    # evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

    # # Step learning rate
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def stepLR(trainer):
    #     if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
    #         scheduler.step(trainer.state.output['train_loss'])#TODO: NOTE: DEBUGGING.... Fix this...
    #     else:
    #         scheduler.step()

    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(
            f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Classifier Loss: {trainer.state.output['train_loss']:.3f} Accuracy: {trainer.state.output['train_accuracy']:.3f} " +
            f"Discriminator: Loss: {trainer.state.output['dom_loss']:.3f} Accuracy: {trainer.state.output['dom_accuracy']:.3f}",
            end='')

    # Log training metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(trainer):
        metrics = evaluator.run(train_loader).metrics
        for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
        if verbose: print(
            f"\nTraining Results - Epoch: {trainer.state.epoch} Classifier: loss: {metrics['train_loss']:.4f} accuracy: {metrics['train_accuracy']:.4f} Discriminator: loss: {metrics['dom_loss']:.4f} accuracy: {metrics['dom_accuracy']:.4f}")

    # Log validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer):
        metrics = evaluator.run(val_loader).metrics
        for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
        if verbose: print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Classifier loss: {metrics['train_loss']:.4f} accuracy: {metrics['train_accuracy']:.4f} Discriminator: loss: {metrics['dom_loss']:.4f} accuracy: {metrics['dom_accuracy']:.4f}")

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)

    # Save models
    torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_model')) #NOTE: Save to cpu state so you can test more easily.
    torch.save(classifier.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_classifier'))
    torch.save(discriminator.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_discriminator'))

    # Create training/validation loss plot
    f_loss = plt.figure()
    plt.subplot()
    plt.title('Loss per epoch')
    plt.plot(logs['train']['train_loss'],'-',color='orange',label="classifier training")
    plt.plot(logs['val']['train_loss'],'-',color='red',label="classifier validation")
    plt.plot(logs['train']['dom_loss'],'--',color='orange',label="discriminator training")
    plt.plot(logs['val']['dom_loss'],'--',color='red',label="discriminator validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    f_loss.savefig(os.path.join(log_dir,'training_loss.png'))

    # Create training/validation accuracy plot
    f_acc = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['train_accuracy'],'-',color='blue',label="classifier training")
    plt.plot(logs['val']['train_accuracy'],'-',color='purple',label="classifier validation")
    plt.plot(logs['train']['dom_accuracy'],'--',color='blue',label="discriminator training")
    plt.plot(logs['val']['dom_accuracy'],'--',color='purple',label="discriminator validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f_acc.savefig(os.path.join(log_dir,'training_accuracy.png'))

    # Run MLFlow routines if requested
    if 'mlflow' in config.keys() and config['mlflow']:

        # Start MLFlow run
        mlflow.start_run()#NOTE: ADDED 10/17/22
        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))

        # Create MLFlow logger
        tracking_uri = config['tracking_uri'] if 'tracking_uri' in config.keys() else '' #TODO: Make this an option
        mlflow_logger = MLflowLogger(tracking_uri=tracking_uri)

        # Log experiment parameters:                                                                                                                                                                                                            
        mlflow.set_tag("trial_number",config['trial_number'] if 'trial_number' in config.keys() else -1)#DEBUGGING ADDED                                                                                                                                                                                        
        mlflow_logger.log_params({
            "seed": config['seed'] if 'seed' in config.keys() else -1,                                                                                                                                                                                                                    
            "batch_size": batch_size,
            "model": model.__class__.__name__,

            "pytorch version": torch.__version__,
            "ignite version": ignite.__version__,
            "cuda version": torch.version.cuda,
            "device name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(-1),
            "trial number": config['trial_number'] if 'trial_number' in config.keys() else -1
        })

        # Attach the logger to the trainer to log training loss at each iteration
        mlflow_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda output : output['loss']
        )

        # Attach the logger to the evaluator on the training dataset after each epoch
        mlflow_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_step_from_engine(trainer),
        )

        # Attach the logger to the evaluator on the validation dataset after each epoch
        mlflow_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_step_from_engine(trainer)
        )

        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
        mlflow_logger.attach_opt_params_handler(
            trainer,
            event_name=Events.ITERATION_STARTED,
            optimizer=optimizer,
            param_name='lr'  # optional
        )
        
        # Save model in MLFlow format
        mlflow_logger.pytorch.log_model(model,model_name)
        
        mlflow.log_figure(f_loss, 'training_loss.png')
        mlflow.log_figure(f_acc,  'training_accuracy.png')
        
        mlflow.end_run()

    return logs
    
def evaluate(
    model,
    device,
    test_loader=None,
    dataset='',
    prefix='',
    max_events=0,
    verbose=True
    ):
    """
    Arguments
    ---------
    model : torch.nn.Module, required
    device : string, required
    test_loader : dgl.dataloading.GraphDataLoader, optional
        Default : None
    dataset : string, optional
        Default : ''
    prefix : string, optional
        Default : ''
    max_events : int, optional
        Default : 0
    verbose : boolean, optional
        Default : True

    Returns
    -------
    Tuple containing test accuracy, softmax of model predictions,
    model predictions as integer labels, dataset labels for correctly
    identified graphs, dataset labels for incorrectly identified graphs

    Description
    -----------
    Run model on test data from a dataset or dataloader.
    """

    # Get model
    model.eval()
    model = model.to(device)

    # Get test dataset
    ds = GraphDataset(prefix+dataset) if test_loader is None else test_loader.dataset #NOTE: Make sure this is copied into ~/.dgl folder if prefix is not specified.
    if test_loader is None:
        ngraphs = min(len(ds),max_events) if max_events>0 else len(ds)
        ds      = Subset(ds,range(ngraphs))

    # Get test graphs and labels
    graphs = dgl.batch(ds.dataset.graphs[ds.indices.start:ds.indices.stop]).to(device) #TODO: Figure out nicer way to use subset
    labels = (ds.dataset.labels[ds.indices.start:ds.indices.stop,0].clone().detach().float().view(-1, 1).to(device)
             if len(np.shape(ds.dataset.labels))==2
             else ds.dataset.labels[ds.indices.start:ds.indices.stop].clone().detach().float().view(-1, 1).to(device)) #IMPORTANT: keep .view() here

    # Get predictions on test dataset
    prediction = model(graphs)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
    test_acc   = (labels == argmax_Y.float()).sum().item() / len(labels)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(test_acc * 100))

    # Copy arrays back to CPU
    labels   = labels.cpu()
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    # Split decisions into true and false arrays
    decisions_true  = ma.array(ds.dataset.labels[ds.indices.start:ds.indices.stop].clone().detach().float(),
                                mask=~(torch.squeeze(argmax_Y) == ds.dataset.labels[ds.indices.start:ds.indices.stop].clone().detach().float()))
    decisions_false = ma.array(ds.dataset.labels[ds.indices.start:ds.indices.stop].clone().detach().float(),
                                mask=~(torch.squeeze(argmax_Y) != ds.dataset.labels[ds.indices.start:ds.indices.stop].clone().detach().float()))

    return test_acc, probs_Y, argmax_Y, decisions_true, decisions_false

def evaluateOnData(
    model,
    device,
    dataset='',
    prefix='',
    max_events=0,
    ):

    """
    Arguments
    ---------
    model : torch.nn.Module, required
    device : torch.device, required
    dataset : string, optional
        Default : ''
    prefix : string, optional
        Default : ''
    max_events : int, optional
        Default : 0

    Returns
    -------
    Tuple of softmax of model predictions, model predictions
    as integer labels, and corresponding dataset labels

    Description
    -----------
    Run model on unlabelled test data from a dataset or dataloader.
    """

    # Get model
    model.eval()
    model = model.to(device)

    # Load dataset
    ds      = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    ngraphs = min(len(ds),max_events) if max_events>0 else len(ds)
    ds      = Subset(ds,range(ngraphs))

    # Get prediction on dataset
    graphs     = dgl.batch(ds.dataset.graphs[ds.indices.start:ds.indices.stop]).to(device) #TODO: Figure out nicer way to use subset
    prediction = model(graphs)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)

    # Copy arrays back to CPU
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    # Get dataset labels
    labels = ma.array(ds.dataset.labels[ds.indices.start:ds.indices.stop].clone().detach().float())

    return probs_Y, argmax_Y, labels
