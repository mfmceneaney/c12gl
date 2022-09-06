#--------------------------------------------------#
# Description: Utility functions and classes for 
#   training and evaluating DGL GNNs.
# Author: Matthew McEneaney
#--------------------------------------------------#

# DGL Graph Learning Imports
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_info, load_info, Subset

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

    Description
    -----------
    Get info about a graph dataset without loading the entire dataset.
    """

    # Load training data
    train_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    num_labels = train_dataset.num_labels
    node_feature_dim = train_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = train_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0
    train_dataset.load()
    train_dataset = Subset(train_dataset,range(1))

    return num_labels, node_feature_dim, edge_feature_dim

def loadGraphDataset(
    dataset="",
    prefix="",
    key="data",
    ekey="",
    split=0.75,
    max_events=1e5,
    indices=None,
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
    indices : tuple, optional
        Tuple of start and stop indices to use
    key : string, optional
        Default : "data".
    ekey : string, optional
        Default : "".
    split : float, optional
        Default : 0.75.
    max_events : int, optional
        Default : 1e5.
    indices : tuple, optional
        Default : None.
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
    train_loader : dgl.GraphDataLoader
        Dataloader for training data
    val_loader : dgl.GraphDataLoader
        Dataloader for validation data
    eval_loader : dgl.GraphDataLoader
        Dataloader for evaluation data, only returned if >3 indices specified
    num_labels : int
        Number of classification labels for dataset
    node_feature_dim : int
        Length of tensors in graph node data
    edge_feature_dim : int
        Length of tensors in graph edge data

    Description
    -----------
    Load a graph dataset into training and validation loaders based on split fraction.
    """

    # Load training data
    this_dataset = GraphDataset(prefix+dataset) # Make sure this is copied into ~/.dgl folder
    this_dataset.load()
    num_labels = this_dataset.num_labels
    node_feature_dim = this_dataset.graphs[0].ndata[key].shape[-1]  if  key != '' else 0
    edge_feature_dim = this_dataset.graphs[0].edata[ekey].shape[-1] if ekey != '' else 0

    # Shuffle entire dataset
    if shuffle: this_dataset.shuffle() #TODO: Make the shuffling to dataloading non-biased???

    # Get training subset
    if indices is not None:
        if len(indices)<3: raise IndexError("Length of indices argument must be >=3.")
        if (indices[0]>=len(this_dataset) or indices[1]>=len(this_dataset)): raise IndexError("First or middle index cannot be greater than length of dataset.")
        if indices[0]>indices[1] or indices[1]>indices[2] or (len(indices)>3 and indices[2]>indices[3]): raise IndexError("Make sure indices are in ascending order left to right.")
    index = int(min(len(this_dataset),max_events)*split)
    train_indices = range(index) if indices is None else range(indices[0],int(min(len(this_dataset),indices[1])))
    train_dataset = Subset(this_dataset,train_indices)

    # Create training dataloader
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    # Load validation data
    index2 = int(min(len(this_dataset),max_events))
    val_indices = range(index,index2) if indices is None else range(indices[1],int(min(len(this_dataset),indices[2])))
    val_dataset = Subset(this_dataset,val_indices)

    # Create testing dataloader
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)

    if indices is not None and len(indices)>=4:

        # Load validation data
        eval_indices = range(indices[2],last_index) if indices is None else range(indices[2],int(min(len(this_dataset),indices[3])))
        eval_dataset = Subset(this_dataset,eval_indices)

        # Create testing dataloader
        eval_loader = GraphDataLoader(
            eval_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers)

        return train_loader, val_loader, eval_loader, num_labels, node_feature_dim, edge_feature_dim
    else:
        return train_loader, val_loader, num_labels, node_feature_dim, edge_feature_dim

#------------------------- Classes -------------------------#

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
        mat_path = os.path.join(self.raw_path,self.mode+'_dgl_graph.bin')
        #NOTE: process data to a list of graphs and a list of labels
        if self.inGraphs is not None and self.inLabels is not None:
            self.graphs, self.labels = self.inGraphs, self.inLabels #DEBUGGING: COMMENTED OUT: torch.LongTensor(self.inLabels)
        else:
            self.graphs, self.labels = load_graphs(mat_path)

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
