#--------------------------------------------------#
# Description: Main for DAGNN routine.
# Author: Matthew McEneaney
#--------------------------------------------------#

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# Plotting Imports
import matplotlib.pyplot as plt

# Utility Imports
import argparse, os

# Local Imports
from c12gl.utils import load_graph_dataset, train_dagnn#, evaluate_dagnn
from c12gl.models import GIN, MLP

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch Domain Adversarial GIN for graph classification')
    
    # Basic training options
    parser.add_argument('--dataset', type=str, default="dataset",
                        help='name of dataset (default: dataset)') #NOTE: Needs to be in ~/.dgl or specify prefix
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for where dataset is stored (default: ~/.dgl/)')
    parser.add_argument('--split', type=float, default=0.75,
                        help='Fraction of dataset to use for evaluation (default: 0.75)')
    parser.add_argument('--max_events', type=float, default=1e5,
                        help='Max number of train/val events to use (default: 1e5)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--nworkers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    parser.add_argument('--batch', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial earning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum number of epochs to train (default: 30)')

    # LR Scheduler options
    parser.add_argument('--step', type=int, default=-1,
                        help='Learning rate step size (default: -1 for ReduceLROnPlateau, 0 uses ExponentialLR, >0 uses StepLR)')
    parser.add_argument('--gamma', type=float, default=0.63,
                        help='Learning rate reduction factor (default: 0.63)')
    parser.add_argument('--thresh', type=float, default=1e-4,
                        help='Minimum change threshold for reducing lr on plateau (default: 1e-4)')

    # Early stopping options
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Minimum change threshold for early stopping (default: 0.0)')
    parser.add_argument('--cumulative_delta', action='store_true',
                        help='Use cumulative change since last patience reset as opposed to last event (default: false)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to wait for early stopping (default: 10)')

    # Model parameter options
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of model layers (default: 2)')
    parser.add_argument('--nmlp', type=int, default=3,
                        help='Number of output MLP layers (default: 3)')
    parser.add_argument('--hdim', type=int, default=64,
                        help='Number of hidden dimensions in model (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='Dropout rate for final layer (default: 0.8)')
    parser.add_argument('--gpooling', type=str, default="max", choices=["sum", "average"],
                        help='Pooling type over entire graph: sum or average')
    parser.add_argument('--npooling', type=str, default="max", choices=["sum", "average", "max"],
                        help='Pooling type over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')

    # Output options
    parser.add_argument('--log', type=str, default='logs/',
                        help='Log directory for histograms (default: logs/)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval for training and validation metrics (default: 10)')
    parser.add_argument('--save_path', type=str, default='model',
                        help='Name for file in which to save model (default: model)')
    parser.add_argument('--verbose', action="store_true",
                                    help='Print messages and graphs')

    # Parse arguments
    args = parser.parse_args()

    # Set up and seed devices
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Setup data and model
    train_loader, val_loader, nclasses, nfeatures_node, nfeatures_edge = load_graph_dataset(dataset=args.dataset, prefix=args.prefix, 
                                                    split=args.split, max_events=args.max_events,
                                                    num_workers=args.nworkers, batch_size=args.batch)

    # Create model
    nfeatures = nfeatures_node
    model = GIN(args.nlayers, args.nmlp, nfeatures,
            args.hdim, nclasses, args.dropout, args.learn_eps, args.npooling,
            args.gpooling).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create lr scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=args.patience,
        threshold=args.thresh, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=args.verbose)
    if args.step==0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma, last_epoch=-1, verbose=args.verbose)
    if args.step>0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma, verbose=args.verbose)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Setup log directory
    try: os.makedirs(args.log)
    except FileExistsError: print('Log directory: ',args.log,' already exists.')

    # Train model
    train_dagnn(
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        args.patience,
        args.min_delta,
        args.cumulative_delta,
        criterion,
        args.epochs,
        dataset=args.dataset,
        prefix=args.prefix,
        log_interval=args.log_interval,
        log_dir=args.log,
        save_path=args.save_path,
        verbose=args.verbose
        )
    
    if args.verbose: plt.show()

if __name__ == '__main__':

    main()
