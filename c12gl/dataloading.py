#--------------------------------------------------#
# Description: Utility functions and classes for 
#   training and evaluating DGL GNNs.
# Author: Matthew McEneaney
#--------------------------------------------------#

# Array Imports
import numpy as np

# DGL Graph Learning Imports
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info, Subset

# PyTorch Imports
import torch

# Utility Imports
import os.path as osp

#------------------------- Functions -------------------------#
# getGraphDatasetInfo
# loadGraphDataset

def getGraphDatasetInfo(
    dataset="",
    prefix="",
    key="data",
    ekey=""
    ):

    """
    Parameters
    ----------
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    key : str, optional
        Default : "data"
    ekey : str, optional
        Default : ""

    Returns
    -------
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Description
    -----------
    Get info about a graph dataset without loading the entire dataset.
    """

    # Load training data
    train_dataset = GraphDataset(prefix+dataset) #NOTE: Make sure this is copied into ~/.dgl folder if prefix not specified.
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = train_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0

    return num_labels, node_feature_dim, edge_feature_dim

def loadGraphDataset(
    dataset='',
    prefix='',
    key='data',
    ekey='',
    split=(,),
    max_events=0,
    batch_size=64,
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
        Default : ''
    prefix : string, optional
        Default : ''
    key : string, optional
        Default : 'data'
    ekey : string, optional
        Default : ''
    split : tuple, optional
        Default : (,)
    max_events : int, optional
        Default : 0
    batch_size : int, optional
        Default : 64
    drop_last : bool, optional
        Default : False
    shuffle : bool, optional
        Default : True
    num_workers : int, optional
        Default : 0
    pin_memory : bool, optional
        Default : True
    verbose : bool, optional
        Default : True

    Returns
    -------
    train_loader : dgl.GraphDataLoader
        Dataloader for training data
    val_loader : dgl.GraphDataLoader
        Dataloader for validation data
    test_loader : dgl.GraphDataLoader
        Dataloader for testing data, only returned if additional fraction specified in split argument
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Description
    -----------
    Load a graph dataset into training and validation loaders based on split fractions.
    """
    # Check normalization of split
    if len(split)>1 and np.sum(split)!=1.0: raise ValueError('Split fractions should add to one if more than one specified.')

    # Load training data
    ds = GraphDataset(prefix+dataset) #NOTE: Make sure this is copied into ~/.dgl folder if prefix not specified.
    num_labels       = ds.num_labels
    node_feature_dim = ds.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = ds.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0
    ngraphs = min(len(ds),max_events)

    # Shuffle entire dataset
    if shuffle: this_dataset.shuffle() #TODO: Make the shuffling to dataloading non-biased?

    # Get training subset
    index1 = int(ngraphs*(split[0] if len(split)>0 else 1))
    train_ds = Subset(ds,range(index1))

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    if len(split)<1: return train_loader, num_labels, node_feature_dim, edge_feature_dim

    # Get validation subset
    index2 = int(ngraphs*(np.sum(split[:1]) if len(split)>1 else 1)) #NOTE: np.sum is important here!
    val_ds = Subset(ds,range(index1,index2))

    # Create validation dataloader
    val_loader = GraphDataLoader(
        val_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    if len(split)<=2: return train_loader, val_loader, num_labels, node_feature_dim, edge_feature_dim

    # Get testing subset
    test_ds = Subset(ds,range(index2,ngraphs))

    # Create testing dataloader
    test_loader = GraphDataLoader(
        test_ds,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return train_loader, val_loader, test_loader num_labels, node_feature_dim, edge_feature_dim

#------------------------- Classes -------------------------#
# GraphDataset

class GraphDataset(DGLDataset):

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
        inLabels= : Tensor, optional
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
        mat_path = osp.join(self.raw_path,self.mode+'_dgl_graph.bin')
        #NOTE: process data to a list of graphs and a list of labels
        if self.inGraphs is not None and self.inLabels is not None:
            self.graphs, self.labels = self.inGraphs, self.inLabels #DEBUGGING: COMMENTED OUT: torch.LongTensor(self.inLabels)
        elif not self.has_cache(): self.graphs, self.labels = [], []
        else:
            self.graphs, self.labels = load_graphs(mat_path)
    
    #----- ADDED -----#
    def extend(self,inLabels,inGraphs):
        """
        Parameters
        ----------
        inLabels : torch.tensor
        inGraphs : list(dgl.graph)

        Description
        -----------
        Adds additional labels and graphs to existing dataset and saves to file.
        """
        self.labels = torch.cat(
                                (
                                    self.labels['labels'] if type(self.labels)==dict
                                    else self.labels if type(self.labels)==torch.Tensor
                                    else torch.tensor(self.labels),
                                    inLabels if type(inLabels)==torch.Tensor
                                    else torch.tensor(inLabels)
                                ),
                                dim=0
                                )
        self.graphs.extend(inGraphs)
        self.save()

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save(self):
        # check that graphs and labels exist
        if len(self.graphs)<=0:
            if self.verbose: print("No graphs to save")
            return

        # save graphs and labels
        graph_path = osp.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = osp.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = osp.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = osp.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = osp.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = osp.join(self.save_path, self.mode + '_info.pkl')
        return osp.exists(graph_path) and osp.exists(info_path)

    def shuffle(self):
        """
        Randomly shuffle dataset graphs and labels.
        """
        indices = np.array([i for i in range(len(self.graphs))])
        np.random.shuffle(indices) #NOTE: In-place method
        self.labels = torch.stack([self.labels[i] for i in indices]) #NOTE: Don't use torch.tensor([some python list]) since that only works for 1D lists.
        self.graphs = [self.graphs[i] for i in indices]
    
    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.num_classes
