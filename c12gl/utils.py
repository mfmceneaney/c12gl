#--------------------------------------------------#
# Description: Utility functions and classes for 
#   training and evaluating DGL GNNs.
# Author: Matthew McEneaney
#--------------------------------------------------#

# ML Imports
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
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
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import global_step_from_engine, EarlyStopping

# Utility Imports
import datetime, os, itertools

# Local Imports
from dataloading import getGraphDatasetInfo, loadGraphDataset, GraphDataset

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
    args,
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
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
    args : str, required
    model : str, required
    device : str, required
    train_loader : str, required
    val_loader : str, required
    optimizer : str, required
    scheduler : str, required
    criterion : str, required
    max_epochs : str, required
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

    Description
    -----------
    Train a GNN using a basic supervised learning approach.
    """

    # Show model if requested
    if verbose: print(model)

    # Logs for matplotlib plots
    logs={'train':{'loss':[],'accuracy':[]}, 'val':{'loss':[],'accuracy':[]}}

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

    # Create evaluator
    evaluator = Engine(val_step)

    # Add evaluation metrics
    accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred_preprocessed'], x['y']])
    accuracy_.attach(evaluator, 'accuracy')
    loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
    loss_.attach(evaluator, 'loss')

    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    handler = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        cumulative_delta=args.cumulative_delta,
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
            scheduler.step(trainer.state.output['loss'])#TODO: NOTE: DEBUGGING.... Fix this...
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

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+"_weights")) #NOTE: Save to cpu state so you can test more easily.
   
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
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+'.png'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+'.png'))

    return logs

def trainDA(
    args,
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
    train_criterion,
    dom_criterion,
    alpha,
    max_epochs,
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True
    ):
    #TODO: GET RID OF ARGS ARGUMENT??!?!!
    """
    Parameters
    ----------
    args : str, required
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
    train_criterion : str, required
    dom_criterion : str, required
    alpha : function, required
    max_epochs : int, required
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

    Description
    -----------
    Train a GNN using a Domain Adversarial approach.
    """

    # Show model if requested
    if verbose: print(model)

    # Logs for matplotlib plots
    logs={'train':{'train_loss':[],'train_accuracy':[],'dom_loss':[],'dom_accuracy':[]},
            'val':{'train_loss':[],'train_accuracy':[],'dom_loss':[],'dom_accuracy':[]}}

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
        dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
        dom_loss = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].
        discriminator.zero_grad()
        dom_loss.backward()
        discriminator_optimizer.step()
        
        # Step the classifier on training data
        train_y = classifier(h[:nLabelled]) #NOTE: Only train on labelled (i.e., training) data, not domain data.
        dom_y = discriminator(h)
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
        train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
        train_probs_y = torch.softmax(train_y, 1)
        train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

        # Apply softmax and get accuracy on domain data
        dom_true_y = dom_labels.clone().detach().float().view(-1, 1) #NOTE: Activation should already be a part of the discriminator
        dom_argmax_y = torch.max(dom_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
        dom_acc = (dom_true_y == dom_argmax_y.float()).sum().item() / len(dom_true_y)

        return {
                'alpha': alpha,
                'train_y': train_y, #CLASSIFIER OUTPUT
                'train_probs_y': train_probs_y,
                'train_true_y': train_labels, #NOTE: Need this for some reason?
                'train_argmax_y': train_argmax_y,
                'train_loss': train_loss.detach().item(),
                'train_accuracy': train_acc,
                'dom_y': dom_y, #DISCRIMINATOR OUTPUT
                'dom_true_y': dom_labels, #NOTE: Need this for some reason?
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
            dom_labels = torch.cat([torch.ones(nLabelled,dtype=torch.long), torch.zeros(nUnlabelled,dtype=torch.long)], dim=0).to(device) #NOTE: Make sure domain label lengths match actual batches at the end.
            dom_loss = dom_criterion(dom_y, dom_labels) #NOTE: Using activation function like nn.Sigmoid() at end of model is important since the predictions need to be in [0,1].
            
            # Step the classifier on training data
            train_y = classifier(h[:nLabelled]) #NOTE: Only evaluate on labelled (i.e., training) data, not domain data.
            dom_y = discriminator(h)
            train_loss = train_criterion(train_y, train_labels)
            dom_loss   = dom_criterion(dom_y, dom_labels) #NOTE: Using activation like nn.Sigmoid() on discriminator is important since the predictions need to be in [0,1].

            # Get total loss using lambda coefficient for epoch
            tot_loss = train_loss - alpha * dom_loss

            # Apply softmax and get accuracy on training data
            train_true_y = train_labels.clone().detach().float().view(-1, 1) #NOTE: Labels for cross entropy loss have to be (N) shaped if input is (N,C) shaped.
            train_probs_y = torch.softmax(train_y, 1)
            train_argmax_y = torch.max(train_probs_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            train_acc = (train_true_y == train_argmax_y.float()).sum().item() / len(train_true_y)

            # Apply softmax and get accuracy on domain data
            dom_true_y = dom_labels.clone().detach().float().view(-1, 1) #NOTE: Activation should already be a part of the discriminator
            dom_argmax_y = torch.max(dom_y, 1)[1].view(-1, 1) #TODO: Could set limit for classification? something like np.where(arg_max_Y>limit)
            dom_acc = (dom_true_y == dom_argmax_y.float()).sum().item() / len(dom_true_y)

        return {
                'alpha': alpha,
                'train_y': train_y, #CLASSIFIER OUTPUT
                'train_probs_y': train_probs_y,
                'train_true_y': train_labels, #NOTE: Need this for some reason?
                'train_argmax_y': train_argmax_y,
                'train_loss': train_loss.detach().item(),
                'train_accuracy': train_acc,
                'dom_y': dom_y, #DISCRIMINATOR OUTPUT
                'dom_true_y': dom_labels, #NOTE: Need this for some reason?
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

    # Add training metrics for discriminator
    dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']])
    dom_accuracy.attach(trainer, 'dom_accuracy')
    dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_y'], x['dom_true_y']])
    dom_loss.attach(trainer, 'dom_loss')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add validation metrics for classifier
    _train_accuracy  = Accuracy(output_transform=lambda x: [x['train_probs_y'], x['train_true_y']])
    _train_accuracy.attach(evaluator, 'train_accuracy')
    _train_loss      = Loss(train_criterion,output_transform=lambda x: [x['train_y'], x['train_true_y']])
    _train_loss.attach(evaluator, 'train_loss')

    # Add validation metrics for discriminator
    _dom_accuracy  = Accuracy(output_transform=lambda x: [x['dom_argmax_y'], x['dom_true_y']])
    _dom_accuracy.attach(evaluator, 'dom_accuracy')
    _dom_loss      = Loss(dom_criterion,output_transform=lambda x: [x['dom_y'], x['dom_true_y']])
    _dom_loss.attach(evaluator, 'dom_loss')

    # Set up early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['train_loss']
        return -val_loss

    handler = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        cumulative_delta=args.cumulative_delta,
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
        if verbose: print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Classifier loss: {metrics['train_loss']:.4f} accuracy: {metrics['train_accuracy']:.4f} Discriminator: loss: {metrics['dom_loss']:.4f} accuracy: {metrics['dom_accuracy']:.4f}")

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    if save_path!="":
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_model_weights')) #NOTE: Save to cpu state so you can test more easily.
        torch.save(classifier.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_classifier_weights'))
        torch.save(discriminator.to('cpu').state_dict(), os.path.join(log_dir,save_path+'_discriminator_weights'))

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
    f.savefig(os.path.join(log_dir,'training_metrics_loss_'+datetime.datetime.now().strftime("%F")+'.png'))

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
    f.savefig(os.path.join(log_dir,'training_metrics_acc_'+datetime.datetime.now().strftime("%F")+'.png'))

    return logs
    
def evaluate(
    model,
    device,
    eval_loader=None,
    dataset="",
    prefix="",
    split=1.0,
    max_events=1e20,
    log_dir="logs/",
    verbose=True
    ):
    """
    Arguments
    ---------
    model : dgl.nn.model
    device : string
    eval_loader : dgl.dataloading.GraphDataLoader, optional
    dataset : string, optional
    prefix : string, optional
    split : float, optional
    max_events : int, optional
    log_dir : string, optional
    verbose : boolean, optional

    Returns
    -------
    Tuple containing test accuracy, model predictions as integer labels,
    dataset labels for correctly identified graphs, dataset labels for incorrectly
    identified graphs

    Description
    -----------
    Run model on test data from a dataset or dataloader.
    """

    # Load validation data
    test_dataset = GraphDataset(prefix+dataset) if eval_loader is None else eval_loader.dataset # Make sure this is copied into ~/.dgl folder
    if eval_loader is None:
        test_dataset.load()
        test_dataset = Subset(test_dataset,range(int(min(len(test_dataset),max_events)*split)))

    model.eval()
    model      = model.to(device)

    test_bg    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop]) #TODO: Figure out nicer way to use subset
    test_Y     = test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float().view(-1, 1) #IMPORTANT: keep .view() here
    test_bg    = test_bg.to(device)
    test_Y     = test_Y.to(device)

    prediction = model(test_bg)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)
    test_acc = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    if verbose: print('Accuracy of predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    # Copy arrays back to CPU
    test_Y   = test_Y.cpu()
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    # Get false-positive true-negatives and vice versa
    decisions_true  = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop].clone().detach().float(),
                                mask=~(torch.squeeze(argmax_Y) == test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float()))
    decisions_false = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop].clone().detach().float(),
                                mask=~(torch.squeeze(argmax_Y) != test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop,0].clone().detach().float()))

    return test_acc, argmax_Y, decisions_true, decisions_false

def evaluateOnData(
    model,
    device,
    dataset="",
    prefix="",
    split=1.0,
    log_dir="logs/",
    verbose=True
    ):

    """
    Arguments
    ---------
    model : torch.nn.model
    device : string
    dataset : string, optional
    prefix : string, optional
    split : float, optional
    log_dir : string, optional
    verbose : boolean, optional

    Returns
    -------
    Array of predictions and corresponding dataset labels

    Description
    -----------
    Run model on unlabelled test data from a dataset or dataloader.
    """

    # Load validation data
    test_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    test_dataset.load()
    test_dataset = Subset(test_dataset,range(int(len(test_dataset)*split)))

    model.eval()
    model      = model.to(device)
    test_bg    = dgl.batch(test_dataset.dataset.graphs[test_dataset.indices.start:test_dataset.indices.stop])#TODO: Figure out nicer way to use subset
    test_bg    = test_bg.to(device)
    prediction = model(test_bg)
    probs_Y    = torch.softmax(prediction, 1)
    argmax_Y   = torch.max(probs_Y, 1)[1].view(-1, 1)

    # Copy arrays back to CPU
    probs_Y  = probs_Y.cpu()
    argmax_Y = argmax_Y.cpu()

    # Get dataset labels
    labels = ma.array(test_dataset.dataset.labels[test_dataset.indices.start:test_dataset.indices.stop].clone().detach().float())

    return argmax_Y, labels
