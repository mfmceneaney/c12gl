#--------------------------------------------------#
# Description: Utility functions and classes for 
#   training DGL GNNs.
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
from torch.nn import DataParallel

# PyTorch Ignite Imports
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import global_step_from_engine, EarlyStopping

# Utility Imports
import datetime, os, itertools

#------------------------- Functions -------------------------#
# get_graph_dataset_info
# load_graph_dataset
# train
# train_dagnn

def get_graph_dataset_info(
    dataset="",
    prefix="",
    key="data",
    ekey=""
    ):

    """
    Parameters
    ----------
    dataset : str, optional
        Default : "".
    prefix : str, optional
        Default : "".
    key : str, optional
        Default : "data".
    ekey : str, optional
        Default : "".

    Returns
    -------
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Examples
    --------

    Notes
    -----

    """

    # Load training data
    train_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = train_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0
    train_dataset.load()
    train_dataset = Subset(train_dataset,range(1))

    return num_labels, node_feature_dim, edge_feature_dim

def load_graph_dataset(
    dataset="",
    prefix="",
    key="data",
    ekey="",
    split=0.75,
    max_events=1e5,
    batch_size=1024,
    drop_last=False,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    verbose=True
    ):

    """
    Parameters
    ----------
    dataset : string, optional
        Default : 1024.
    prefix : string, optional
        Default : False.
    key : string, optional
        Default : "data".
    ekey : string, optional
        Default : "".
    split : float, optional
        Default : 0.75.
    max_events : int, optional
        Default : 1e5.
    batch_size : int, optional
        Default : 1024.
    drop_last : bool, optional
        Default : False.
    shuffle : bool, optional
        Default : False.
    num_workers : int, optional
        Default : 0.
    pin_memory : bool, optional
        Default : True.
    verbose : bool, optional
        Default : True

    Returns
    -------
    train_loader : dgl.dataloading.GraphDataLoader
        Dataloader for training data
    val_loader : dgl.dataloading.GraphDataLoader
        Dataloader for validation data
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Examples
    --------

    Notes
    -----
    Load a graph dataset into training and validation loaders based on split fraction.
    """

    # Load training data
    train_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    train_dataset.load()
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = train_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0
    index = int(min(len(train_dataset),max_events)*split)
    train_dataset = Subset(train_dataset,range(index))

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    # Load validation data
    val_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    val_dataset.load()
    val_dataset = Subset(val_dataset,range(index,len(val_dataset)))

    # Create testing dataloader
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return train_loader, val_loader, num_labels, node_feature_dim, edge_feature_dim

def train(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    patience,         #NOTE: Early stopping parameter
    min_delta,        #NOTE: Early stopping parameter
    cumulative_delta, #NOTE: Early stopping parameter
    criterion,
    max_epochs,
    dataset="",
    prefix="",
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True
    ):

    """
    Parameters
    ----------
    model : torch.nn or models.model, required
    device : str, required
    train_loader : dgl.dataloading.GraphDataLoader, required
    val_loader : dgl.dataloading.GraphDataLoader, required
    optimizer : torch.optim., required
    scheduler : torch.optim.lr_scheduler, required
    criterion : torch.nn, required
    max_epochs : int, required
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    log_interval : int, optional
        Default : 10.
    log_dir : str, optional
        Default : "logs/".
    save_path : str, optional
        Default : "model"
    verbose : bool, optional
        Default : True

    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch

    Examples
    --------

    Notes
    -----

    """

    # Make sure log/save directories exist
    try:
        os.makedirs(log_dir+"tb_logs/tmp") #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
    except FileExistsError:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"tb_logs/tmp"))

    # Show model if requested
    if verbose: print(model)

    # Logs for matplotlib plots
    logs={'train':{'loss':[],'accuracy':[],'roc_auc':[]}, 'val':{'loss':[],'accuracy':[],'roc_auc':[]}}

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

        # Get predictions and loss from data and labels
        x, label   = batch
        y = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
        x      = x.to(device)
        y      = y.to(device)
        y_pred = model(x)
        loss   = criterion(y_pred, y)

        # Step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply softmax and get accuracy
        test_Y = y.clone().detach().float().view(-1, 1) 
        probs_Y = torch.softmax(y_pred, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)

        return {
                'y_pred': y_pred,
                'y': y,
                'y_pred_preprocessed': argmax_Y,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.

            # Get predictions and loss from data and labels
            x, label   = batch
            y = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
            x      = x.to(device)
            y      = y.to(device)
            y_pred = model(x)
            loss   = criterion(y_pred, y)

            # Apply softmax and get accuracy
            test_Y = y.clone().detach().float().view(-1, 1) 
            probs_Y = torch.softmax(y_pred, 1)
            argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)

        return {
                'y_pred': y_pred,
                'y': y,
                'y_pred_preprocessed': argmax_Y,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics
    accuracy  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy.attach(trainer, 'accuracy')
    loss      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss.attach(trainer, 'loss')
    roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    roc_auc.attach(trainer,'roc_auc')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add evaluation metrics
    accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy_.attach(evaluator, 'accuracy')
    loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss_.attach(evaluator, 'loss')
    roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    roc_auc_.attach(evaluator,'roc_auc')

    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss'] #TODO: Select manually which metric?
        return -val_loss

    handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        cumulative_delta=cumulative_delta,
        score_function=score_function,
        trainer=trainer
        )
    evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

    # Step learning rate #NOTE: DEBUGGING: TODO: Replace above...
    @trainer.on(Events.EPOCH_COMPLETED)
    def stepLR(trainer):
        if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(trainer.state.output['loss'])#TODO: NOTE: DEBUGGING. Fix this...
        else:
            scheduler.step()

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

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training_by_iteration",
        output_transform=lambda x: x["loss"]
    )
        
    # Attach the logger to the evaluator on the training dataset and log Loss, Accuracy metrics after each epoch
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["loss","accuracy","roc_auc"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log Loss, Accuracy metrics after
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["loss","accuracy","roc_auc"],
        global_step_transform=global_step_from_engine(evaluator)
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name='lr'  # optional
    )

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close() #IMPORTANT!
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+"_state_dict")) #NOTE: Save to cpu state so you can test more easily.
        # torch.save(model.to('cpu'), os.path.join(log_dir,save_path)) #NOTE: Save to cpu state so you can test more easily.
   
    # Create training/validation loss plot
    f = plt.figure()
    plt.subplot()
    plt.title('Loss per epoch')
    plt.plot(logs['train']['loss'],label="training")
    plt.plot(logs['val']['loss'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    return logs

def train_dagnn(
    model,
    classifier,
    discriminator,
    device,
    train_loader,
    val_loader,
    dom_train_loader,
    dom_val_loader,
    model_optimizer,
    classifier_optimizer,
    discriminator_optimizer,
    scheduler,
    patience,
    min_delta,
    cumulative_delta,
    train_criterion,
    dom_criterion,
    lambda_function,
    max_epochs,
    dataset="",
    prefix="",
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True
    ):

    """
    Parameters
    ----------
    model : str, required
    classifier : str, required
    discriminator : str, required
    device : str, required
    train_loader : str, required
    val_loader : str, required
    dom_train_loader : str, required
    dom_val_loader : str, required
    model_optimizer : str, required
    classifier_optimizer : str, required
    discriminator_optimizer : str, required
    scheduler : str, required
    patience : int, required
    min_delta : float, required
    cumulative_delta : float, required
    train_criterion : str, required
    dom_criterion : str, required
    lambda_function : callable, required
    max_epochs : int, required
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    log_interval : int, optional
        Default : 10
    log_dir : str, optional
        Default : "logs/"
    save_path : str, optional
        Default : "model".
    verbose : bool, optional
        Default : True.

    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch

    Examples
    --------

    Notes
    -----

    """

    # Make sure log/save directories exist
    try:
        os.makedirs(log_dir+"tb_logs/tmp") #NOTE: Do NOT use os.path.join() here since it requires that the directory exist.
    except FileExistsError:
        if verbose: print("Could not create directory:",os.path.join(log_dir,"tb_logs/tmp"))

    # Show model if requested
    if verbose: print(model)

    # Logs for matplotlib plots
    logs={'train':{'train_loss':[],'train_accuracy':[],'train_roc_auc':[],'dom_loss':[],'dom_accuracy':[],'dom_roc_auc':[]},
            'val':{'train_loss':[],'train_accuracy':[],'train_roc_auc':[],'dom_loss':[],'dom_accuracy':[],'dom_roc_auc':[]}}

    # Continuously sample target domain data for training and validation
    dom_train_set = itertools.cycle(dom_train_loader)
    dom_val_set   = itertools.cycle(dom_val_loader)

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

       # Get domain data
       tgt = dom_train_set.__next__()[0] #NOTE: This returns [dgl.HeteroGraph,torch.tensor] for graph and labels.
       tgt = tgt.to(device)

        # Get predictions and loss from data and labels
        x, label     = batch
        train_labels = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
        x            = x.to(device)
        train_labels = train_labels.to(device)

        # Concatenate classification data and domain data
        x = dgl.unbatch(x)
        tgt = dgl.unbatch(tgt)
        nLabelled   = len(x)
        nUnlabelled = len(tgt)
        x.extend(tgt)
        x = dgl.batch(x) #NOTE: Training and domain data must have the same schema for this to work.

        # Get hidden representation from model on training and domain data
        h = model(x)
        
        # Step the domain discriminator on training and domain data
        dom_y = discriminator(h.detach())
        dom_labels = torch.cat([torch.ones(nLabelled,1), torch.zeros(nUnlabelled,1)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
        dom_loss = dom_criterion(dom_y, dom_labels)
        discriminator.zero_grad()
        dom_loss.backward()
        discriminator_optimizer.step()
        
        # Step the classifier on training data
        train_y = classifier(h[:nLabelled]) #NOTE: Only train on labelled (i.e., training) data, not domain data.
        dom_y = discriminator(h)
        train_loss = train_criterion(train_y, train_labels)
        dom_loss   = dom_criterion(dom_y, dom_labels)

        # Get total loss using lambda coefficient for epoch
        coeff = lambda_function(engine.state.epoch, max_epochs)
        tot_loss = train_loss - coeff * dom_loss
        
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
        train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
        train_probs_y = torch.softmax(train_y, 1)
        train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

        # Apply softmax and get accuracy on domain data
        dom_true_y = dom_labels.clone().detach().float().view(-1, 1) 
        # dom_probs_y = torch.softmax(dom_y, 1) #NOTE: Activation should already be a part of the discriminator
        dom_argmax_y = torch.max(dom_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        dom_acc = (dom_true_y == dom_argmax_y.float()).sum().item() / len(dom_true_y)

        return {
                'lambda' : coeff,
                'train_y': train_y, #CLASSIFIER OUTPUT
                'train_true_y': train_labels, #NOTE: Need these labels for some reason.
                'train_argmax_y': train_argmax_y,
                'train_loss': train_loss.detach().item(),
                'train_accuracy': train_acc,
                'dom_y': dom_y, #DISCRIMINATOR OUTPUT
                'dom_true_y': dom_labels, #NOTE: Need these labels for some reason.
                'dom_argmax_y': dom_argmax_y,
                'dom_loss': dom_loss.detach().item(),
                'dom_accuracy': dom_acc,
                'tot_loss': tot_loss.detach().item() #TOTAL LOSS
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.
            
            # Get domain data
            tgt = dom_val_set.__next__()[0] #NOTE: This returns [dgl.HeteroGraph,torch.tensor] for graph and labels.
            tgt = tgt.to(device)

            # Get predictions and loss from data and labels
            x, label     = batch
            train_labels = label[:,0].clone().detach().long() #NOTE: This assumes labels is 2D.
            x            = x.to(device)
            train_labels = train_labels.to(device)

            # Concatenate classification data and domain data
            x = dgl.unbatch(x)
            tgt = dgl.unbatch(tgt)
            nLabelled   = len(x)
            nUnlabelled = len(tgt)
            x.extend(tgt)
            x = dgl.batch(x) #NOTE: Training and domain data must have the same schema for this to work.

            # Get hidden representation from model on training and domain data
            h = model(x)
            
            # Step the domain discriminator on training and domain data
            dom_y = discriminator(h.detach())
            dom_labels = torch.cat([torch.ones(nLabelled,1), torch.zeros(nUnlabelled,1)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
            dom_loss = dom_criterion(dom_y, dom_labels)
            
            # Step the classifier on training data
            train_y = classifier(h[:nLabelled]) #NOTE: Only evaluate on labelled (i.e., training) data, not domain data.
            dom_y = discriminator(h)
            train_loss = train_criterion(train_y, train_labels)
            dom_loss   = dom_criterion(dom_y, dom_labels)

            # Get total loss using lambda coefficient for epoch
            coeff = lambda_function(engine.state.epoch, max_epochs)
            tot_loss = train_loss - coeff * dom_loss

            # Apply softmax and get accuracy on training data
            train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
            train_probs_y = torch.softmax(train_y, 1)
            train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

            # Apply softmax and get accuracy on domain data
            dom_true_y = dom_labels.clone().detach().float().view(-1, 1) 
            # dom_probs_y = torch.softmax(dom_y, 1) #NOTE: Activation should already be a part of the discriminator
            dom_argmax_y = torch.max(dom_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            dom_acc = (dom_true_y == dom_argmax_y.float()).sum().item() / len(dom_true_y)

        return {
                'lambda' : coeff,
                'train_y': train_y, #CLASSIFIER OUTPUT
                'train_true_y': train_labels, #NOTE: Need these labels for some reason.
                'train_argmax_y': train_argmax_y,
                'train_loss': train_loss.detach().item(),
                'train_accuracy': train_acc,
                'dom_y': dom_y, #DISCRIMINATOR OUTPUT
                'dom_true_y': dom_labels, #NOTE: Need these labels for some reason.
                'dom_argmax_y': dom_argmax_y,
                'dom_loss': dom_loss.detach().item(),
                'dom_accuracy': dom_acc,
                'tot_loss': tot_loss.detach().item() #TOTAL LOSS
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics for classifier
    train_accuracy  = Accuracy(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']])
    train_accuracy.attach(trainer, 'train_accuracy')
    train_loss      = Loss(train_criterion,output_transform=lambda x: [x['train_y'], x['train_true_y']])
    train_loss.attach(trainer, 'train_loss')
    # train_roc_auc   = ROC_AUC(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # train_roc_auc.attach(trainer,'train_roc_auc')

    # Add training metrics for discriminator
    dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']])
    dom_accuracy.attach(trainer, 'dom_accuracy')
    dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_y'], x['dom_true_y']])
    dom_loss.attach(trainer, 'dom_loss')
    # dom_roc_auc   = ROC_AUC(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # dom_roc_auc.attach(trainer,'dom_roc_auc')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add validation metrics for classifier
    _train_accuracy  = Accuracy(output_transform=lambda x: [x['train_y'], x['train_true_y']])
    _train_accuracy.attach(evaluator, 'train_accuracy')
    _train_loss      = Loss(train_criterion,output_transform=lambda x: [x['train_y'], x['train_true_y']])
    _train_loss.attach(evaluator, 'train_loss')
    # _train_roc_auc   = ROC_AUC(output_transform=lambda x: [x['train_y'], x['train_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # _train_roc_auc.attach(evaluator,'train_roc_auc')

    # Add validation metrics for discriminator
    _dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']])
    _dom_accuracy.attach(evaluator, 'dom_accuracy')
    _dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_y'], x['dom_true_y']])
    _dom_loss.attach(evaluator, 'dom_loss')
    # _dom_roc_auc   = ROC_AUC(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']]) #NOTE: ROC_AUC CURRENTLY NOT WORKING HERE, NOT SURE WHY...
    # _dom_roc_auc.attach(evaluator,'dom_roc_auc')

    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['train_loss'] #TODO: Select manually which metric?
        return -val_loss

    handler = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        cumulative_delta=cumulative_delta,
        score_function=score_function,
        trainer=trainer
        )
    evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(
            f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Classifier Loss: {trainer.state.output['train_loss']:.3f} Accuracy: {trainer.state.output['train_accuracy']:.3f} " +
            f"Discriminator: Loss: {trainer.state.output['dom_loss']:.3f} Accuracy: {trainer.state.output['dom_accuracy']:.3f}",
            end='')

    # Step learning rate
    @trainer.on(Events.EPOCH_COMPLETED)
    def stepLR(trainer):
        if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(trainer.state.output['train_loss'])#TODO: NOTE: DEBUGGING.... Fix this...
        else:
            scheduler.step()

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
        if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg classifier loss: {metrics['train_loss']:.4f} Avg classifier accuracy: {metrics['train_accuracy']:.4f}")

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training_by_iteration",
        output_transform=lambda x: x["train_loss"]
    )
        
    # Attach the logger to the evaluator on the training dataset and log Loss, Accuracy metrics after each epoch
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["train_loss","train_accuracy","train_roc_auc","dom_loss","dom_accuracy","dom_roc_auc"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log Loss, Accuracy metrics after
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["train_loss","train_accuracy","train_roc_auc","dom_loss","dom_accuracy","dom_roc_auc"],
        global_step_transform=global_step_from_engine(evaluator)
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=model_optimizer,
        param_name='lr'  # optional
    )#TODO: Add other learning rates?

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    tb_logger.close() #IMPORTANT!
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_model_weights')) #NOTE: Save to cpu state so you can test more easily.
        torch.save(classifier.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_classifier_weights'))
        torch.save(discriminator.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_discriminator_weights'))
        # torch.save(model.to('cpu'), os.path.join(log_dir,save_path+'_model')) #NOTE: Save to cpu state so you can test more easily.
        # torch.save(classifier.to('cpu'), os.path.join(log_dir,save_path+'_classifier'))
        # torch.save(discriminator.to('cpu'), os.path.join(log_dir,save_path+'_discriminator'))

    # Create training/validation loss plot
    f = plt.figure()
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
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['train_accuracy'],'-',color='blue',label="classifier training")
    plt.plot(logs['val']['train_accuracy'],'-',color='purple',label="classifier validation")
    plt.plot(logs['train']['dom_accuracy'],'--',color='blue',label="discriminator training")
    plt.plot(logs['val']['dom_accuracy'],'--',color='purple',label="discriminator validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+"_"+dataset+"_nEps"+str(max_epochs)+'.png'))

    return logs

#------------------------- Classes -------------------------#
# GraphDataset

class GraphDataset(DGLDataset):

    """
    Attributes
    ----------
    inGraphs : dgl.HeteroGraph
    inLabels : dgl.HeteroGraph
    mode : str
    num_classes : int

    Methods
    -------

    """

    _url = None
    _sha1_str = None
    mode = "mode"

    def __init__(
        self,
        name="dataset",
        dataset=None,
        inGraphs=None,
        inLabels=None,
        raw_dir=None,
        mode="mode",
        url=None,
        force_reload=False,
        verbose=False,
        num_classes=2
        ):

        """
        Parameters
        ----------
        name : str, optional
            Default : "dataset".
        inGraphs : Tensor(dgl.HeteroGraph), optional
            Default : None.
        inLabels : Tensor, optional
            Default : None.
        raw_dir : str, optional
            Default : None.
        mode : str, optional
            Default : "mode".
        url : str, optional
            Default : None.
        force_reload : bool, optional
            Default : False.
        verbose : bool, optional
            Default : False.
        num_classes : int, optional
            Default : 2.

        Examples
        --------

        Notes
        -----
        
        """
        
        self.inGraphs = inGraphs #NOTE: Set these BEFORE calling super.
        self.inLabels = inLabels
        self._url = url
        self.mode = mode
        self.num_classes = num_classes #NOTE: IMPORTANT! You need the self.num_classes variable for the builtin methods of DGLDataset to work!
        super(GraphDataset, self).__init__(name=name,
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose
                                          )
        
        
    def process(self):
        mat_path = os.path.join(self.raw_path,self.mode+'_dgl_graph.bin')
        #NOTE: process data to a list of graphs and a list of labels
        if self.inGraphs is not None and self.inLabels is not None:
            self.graphs, self.labels = self.inGraphs, torch.LongTensor(self.inLabels)
        else:
            self.graphs, self.labels = load_graphs(mat_path)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)

        Notes
        -----
        Get graph and label by index
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """
        Returns
        -------
        Number of graphs in the dataset
        """
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.num_classes
