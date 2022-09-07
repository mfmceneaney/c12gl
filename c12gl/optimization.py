

# Optuna imports
import optuna
from optuna.samplers import TPESampler

# Local Imports
from .models import GIN, Concatenate, SigmoidMLP
from .dataloading import getGraphDatasetInfo, loadGraphDataset, GraphDataset
from .utils import train, trainDA, evaluate, evaluateOnData

def optimizationStudy(
    args,
    objective,
    device=torch.device('cpu'),
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True):

    #----- MAIN PART -----#
    
    # Load or create pruner, sampler, and study
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    sampler = TPESampler() #TODO: Add command line option for selecting different sampler types.
    study = optuna.create_study(storage='sqlite:///'+args.db_path, sampler=sampler,pruner=pruner, study_name=args.study_name, direction="minimize", load_if_exists=True) #TODO: Add options for different SQL programs: Postgre, MySQL, etc.

    # Run optimization
    study.optimize(objective, n_trials=args.ntrials, timeout=args.timeout, gc_after_trial=True) #NOTE: gc_after_trial=True is to avoid OOM errors see https://optuna.readthedocs.io/en/stable/faq.html#out-of-memory-gc-collect
    trial = study.best_trial

    if verbose:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

def objective(trial):

    """
    Arguments
    ---------
    trial : optuna.trial

    Returns
    -------
    metric : float

    Description
    -----------
    Defines a basic objective function for an optuna optimization study
    with a GIN model.
    """

    # Get parameter suggestions for trial
    batch_size = args.batch[0] if args.batch[0] == args.batch[1] else trial.suggest_int("batch_size",args.batch[0],args.batch[1]) 
    nlayers = args.nlayers[0] if args.nlayers[0] == args.nlayers[1] else trial.suggest_int("nlayers",args.nlayers[0],args.nlayers[1])
    nmlp  = args.nmlp[0] if args.nmlp[0] == args.nmlp[1] else trial.suggest_int("nmlp",args.nmlp[0],args.nmlp[1])
    hdim  = args.hdim[0] if args.hdim[0] == args.hdim[1] else trial.suggest_int("hdim",args.hdim[0],args.hdim[1])
    do    = args.dropout[0] if args.dropout[0] == args.dropout[1] else trial.suggest_float("do",args.dropout[0],args.dropout[1])
    lr    = args.lr[0] if args.lr[0] == args.lr[1] else trial.suggest_float("lr",args.lr[0],args.lr[1],log=True)
    step  = args.step[0] if args.step[0] == args.step[1] else trial.suggest_int("step",args.step[0],args.step[1])
    gamma = args.gamma[0] if args.gamma[0] == args.gamma[1] else trial.suggest_float("gamma",args.gamma[0],args.gamma[1])
    max_epochs = args.epochs

    # Setup data and model #NOTE: DO THIS HERE SINCE IT DEPENDS ON BATCH SIZE. #TODO: NOTE HOPEFULLY ALL BELOW WORKS...
    train_dataloader, val_dataloader, eval_loader, nclasses, nfeatures, nfeatures_edge = [None for i in range(6)]
    if len(args.indices)>3:
        train_dataloader, val_dataloader, eval_loader, nclasses, nfeatures, nfeatures_edge = loadGraphDataset(dataset=args.dataset, prefix=args.prefix, 
                                                split=args.split, max_events=args.max_events, indices=args.indices,
                                                num_workers=args.nworkers, batch_size=batch_size) 
    elif len(args.indices)==3:
        train_dataloader, val_dataloader, nclasses, nfeatures, nfeatures_edge = loadGraphDataset(dataset=args.dataset, prefix=args.prefix, 
                                                split=args.split, max_events=args.max_events, indices=args.indices,
                                                num_workers=args.nworkers, batch_size=batch_size)
    else:
        train_dataloader, val_dataloader, nclasses, nfeatures, nfeatures_edge = loadGraphDataset(dataset=args.dataset, prefix=args.prefix, 
                                                split=args.split, max_events=args.max_events,
                                                num_workers=args.nworkers, batch_size=batch_size)

    # Instantiate model, optimizer, scheduler, and loss
    model = GIN(nlayers,nmlp,nfeatures,hdim,nclasses,do,args.learn_eps,args.npooling,args.gpooling).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=args.patience,
        threshold=args.thresh, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=args.verbose)
    if step==0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=args.verbose)
    if step>0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma, verbose=args.verbose)
    criterion = nn.CrossEntropyLoss()

    # Make sure log/save directories exist
    trialdir = 'trial_'+datetime.datetime.now().strftime("%F")+'_'+args.study_name+'_'+str(trial.number+1)
    try:
        os.makedirs(args.log+'/'+trialdir) #NOTE: Do NOT use os.path.join() here since it requires that the directory already exist.
    except FileExistsError:
        if args.verbose: print("Directory exists: ",os.path.join(args.log,trialdir))
    trialdir = os.path.join(args.log,trialdir)

    # Show model if requested
    if args.verbose: print(model)

    # Logs for matplotlib plots
    logs = train(
                args,
                model,
                device,
                train_dataloader,
                val_dataloader,
                optimizer,
                scheduler,
                criterion,
                args.epochs,
                dataset=args.dataset,
                prefix=args.prefix,
                log_dir=trialdir,
                verbose=args.verbose
                )

    # Get testing AUC
    metrics = evaluate(
        model,
        device,
        eval_loader=eval_loader, #TODO: IMPLEMENT THIS
        dataset=args.dataset, 
        prefix=args.prefix,
        split=args.split,
        max_events=args.max_events,
        log_dir=trialdir,
        verbose=True
    )

    return 1.0 - metrics[0] #NOTE: This is so you maximize AUC since can't figure out how to create sqlite3 study with maximization at the moment 8/5/22


def optimizationStudyDA(
    args,
    device=torch.device('cpu'),
    log_interval=10, #NOTE: log_dir='logs/' should end with the slash
    log_dir="logs/",
    save_path="torch_models",
    verbose=True):

def objective(trial):

    # Get parameter suggestions for trial
    alpha = args.alpha[0] if args.alpha[0] == args.alpha[1] else trial.suggest_int("alpha",args.alpha[0],args.alpha[1])
    batch_size = args.batch[0] if args.batch[0] == args.batch[1] else trial.suggest_int("batch_size",args.batch[0],args.batch[1]) 
    nlayers = args.nlayers[0] if args.nlayers[0] == args.nlayers[1] else trial.suggest_int("nlayers",args.nlayers[0],args.nlayers[1])
    nmlp  = args.nmlp[0] if args.nmlp[0] == args.nmlp[1] else trial.suggest_int("nmlp",args.nmlp[0],args.nmlp[1])
    hdim  = args.hdim[0] if args.hdim[0] == args.hdim[1] else trial.suggest_int("hdim",args.hdim[0],args.hdim[1])
    nmlp_head  = args.nmlp_head[0] if args.nmlp_head[0] == args.nmlp_head[1] else trial.suggest_int("nmlp_head",args.nmlp_head[0],args.nmlp_head[1])
    hdim_head  = args.hdim_head[0] if args.hdim_head[0] == args.hdim_head[1] else trial.suggest_int("hdim_head",args.hdim_head[0],args.hdim_head[1])
    do    = args.dropout[0] if args.dropout[0] == args.dropout[1] else trial.suggest_float("do",args.dropout[0],args.dropout[1])
    lr    = args.lr[0] if args.lr[0] == args.lr[1] else trial.suggest_float("lr",args.lr[0],args.lr[1],log=True)
    lr_c  = args.lr_c[0] if args.lr_c[0] == args.lr_c[1] else trial.suggest_float("lr_c",args.lr_c[0],args.lr_c[1],log=True)
    lr_d  = args.lr_d[0] if args.lr_d[0] == args.lr_d[1] else trial.suggest_float("lr_d",args.lr_d[0],args.lr_d[1],log=True)
    step  = args.step[0] if args.step[0] == args.step[1] else trial.suggest_int("step",args.step[0],args.step[1])
    gamma = args.gamma[0] if args.gamma[0] == args.gamma[1] else trial.suggest_float("gamma",args.gamma[0],args.gamma[1])
    max_epochs = args.epochs

    # Setup data and model #NOTE: DO THIS HERE SINCE IT DEPENDS ON BATCH SIZE. #TODO: NOTE HOPEFULLY ALL BELOW WORKS...
    train_dataloader, val_dataloader, eval_loader, dom_train_loader, dom_val_loader, nclasses, nfeatures, nfeatures_edge = [None for i in range(8)]
    if len(args.indices)>3:
        train_loader, val_loader, eval_loader, nclasses, nfeatures_node, nfeatures_edge = loadGraphDataset(dataset=args.dataset, prefix=args.prefix, 
                                                split=args.split, max_events=args.max_events, indices=args.indices,
                                                num_workers=args.nworkers, batch_size=batch_size)

        dom_train_loader, dom_val_loader, dom_nclasses, dom_nfeatures_node, dom_nfeatures_edge = loadGraphDataset(dataset=args.dom_dataset, prefix=args.dom_prefix, 
                                                split=args.split, max_events=args.max_events, indices=args.indices[0:3],
                                                num_workers=args.nworkers, batch_size=batch_size) 
    elif len(args.indices)==3:
        train_loader, val_loader, eval_loader, nclasses, nfeatures_node, nfeatures_edge = loadGraphDataset(dataset=args.dataset, prefix=args.prefix, 
                                                split=args.split, max_events=args.max_events, indices=args.indices,
                                                num_workers=args.nworkers, batch_size=batch_size)

        dom_train_loader, dom_val_loader, dom_nclasses, dom_nfeatures_node, dom_nfeatures_edge = loadGraphDataset(dataset=args.dom_dataset, prefix=args.dom_prefix, 
                                                split=args.split, max_events=args.max_events, indices=args.indices,
                                                num_workers=args.nworkers, batch_size=batch_size)
    else:
        train_loader, val_loader, nclasses, nfeatures_node, nfeatures_edge = loadGraphDataset(dataset=args.dataset, prefix=args.prefix, 
                                                split=args.split, max_events=args.max_events,
                                                num_workers=args.nworkers, batch_size=batch_size)

        dom_train_loader, dom_val_loader, dom_nclasses, dom_nfeatures_node, dom_nfeatures_edge = loadGraphDataset(dataset=args.dom_dataset, prefix=args.dom_prefix, 
                                                split=args.split, max_events=args.max_events,
                                                num_workers=args.nworkers, batch_size=batch_size)

    # Check that # classes and data dimensionality at nodes and edges match between training and domain data
    if nclasses!=dom_nclasses or nfeatures_node!=dom_nfeatures_node or nfeatures_edge!=dom_nfeatures_edge:
        print("*** ERROR *** mismatch between graph structure for domain and training data!")
        print("EXITING...")
        return

    n_domains = 2
    nfeatures = nfeatures_node

    # Create models
    model = GIN(nlayers, nmlp, nfeatures,
            hdim, hdim, do, args.learn_eps, args.npooling,
            args.gpooling).to(device)
    classifier = SigmoidMLP(nmlp_head, hdim, hdim_head, nclasses).to(device)
    discriminator = SigmoidMLP(nmlp_head, hdim, hdim_head, n_domains).to(device)

    # Create optimizers
    model_optimizer = optim.Adam(model.parameters(), lr=lr)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr_c)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d)

    # Create schedulers
    model_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', factor=gamma, patience=args.patience,
        threshold=args.thresh, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=args.verbose)
    if step==0:
        model_scheduler = optim.lr_scheduler.ExponentialLR(model_optimizer, gamma, last_epoch=-1, verbose=args.verbose)
    if step>0:
        model_scheduler = optim.lr_scheduler.StepLR(model_optimizer, step_size=step, gamma=gamma, verbose=args.verbose)

    # Create loss functions
    train_criterion = nn.CrossEntropyLoss()
    dom_criterion   = nn.CrossEntropyLoss()

    # Make sure log/save directories exist
    trialdir = 'trial_'+datetime.datetime.now().strftime("%F")+'_'+args.study_name+'_'+str(trial.number+1)
    try:
        os.makedirs(args.log+'/'+trialdir) #NOTE: Do NOT use os.path.join() here since it requires that the directory already exist.
    except FileExistsError:
        if args.verbose: print("Directory exists: ",os.path.join(args.log,trialdir))
    trialdir = os.path.join(args.log,trialdir)

    # Show model if requested
    if args.verbose: print(model)

    # Logs for matplotlib plots
    logs = trainDA(
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
                        model_scheduler,
                        train_criterion,
                        dom_criterion,
                        alpha,
                        args.epochs,
                        dataset=args.dataset,
                        prefix=args.prefix,
                        log_interval=args.log_interval,
                        log_dir=trialdir,
                        save_path=args.save_path,
                        verbose=args.verbose
                    )

    # Setup data and model
    nclasses, nfeatures, nfeatures_edge = getGraphDatasetInfo(dataset=args.dataset, prefix=args.prefix)
    model_concatenate = Concatenate([ model, classifier])
    model_concatenate.eval()

    # Get testing AUC
    metrics = evaluate(
        model_concatenate,
        device,
        dataset=args.dataset,
        prefix=args.prefix,
        eval_loader=eval_loader,
        split=args.split,
        max_events=args.max_events,
        log_dir=trialdir,
        verbose=True
    )

    return 1.0 - metrics[0] #NOTE: This is so you maximize AUC since can't figure out how to create sqlite3 study with maximization at the moment 8/9/22
