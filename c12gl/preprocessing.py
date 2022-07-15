#--------------------------------------------------#
# Description: Data preprocessing help
# Author: Matthew McEneaney
#--------------------------------------------------#

from __future__ import absolute_import, division, print_function

# NumPy Imports
import numpy as np
import numpy.ma as ma

# Deep Graph Learning Imports
import dgl

# PyTorch Imports
import torch

# File I/O Imports
import uproot
import hipopy.hipopy as hipopy

#TODO: Check method names and attribute names and make sure camelBack vs. _ convention is consistent
#TODO: Check doc strings

#------------------------- Functions: -------------------------#
# normalize, getRingGraph, getWebGraph

def normalize(arr,mean=None,std=None,max=False,index=-2,log=False,inplace=False):
    """
    Parameters
    ----------
    arr : numpy.ma.array, required
        Input 2D ma.array to normalize event by event
    mean : float, optional
        Mean to which to normalize events
    std : float, optional
        Standard deviation to which to normalize events
    max : bool, optional
        Whether to normalize to maximum deviation from event mean
    index : int, optional
        Index along which to normalize, NOTE: Fixed for now.
    log : bool, optional
        Whether to take log before normalization
    inplace : bool, optional
        Whether to modify input array in place

    Returns
    -------
    newarr : numpy.ma.array
        Event normalized 2D array

    Description
    -----------
    Normalizes input array to difference from event mean divided by
    standard deviation or maximum deviation. Mean and standard deviation may also be 
    statically specified. Optionally takes log of distribution
    before normalization.
    """
    newarr = arr.copy() if not inplace else arr #TODO: Fix this so it's along an arbitrary index.

    # Custom mean and stddev case
    if mean is not None and std is not None:
        if std == 0.0: raise ValueError
        for idx in range(len(newarr)):
            if len(newarr[idx]==0): continue
            if log: newarr[idx] = np.log(idx)
            newarr[idx] = (newarr[idx]-mean)/std #TODO: COMBOS OF STATIC MEAN DYNAMIC STD ETC.

    # Normalize maximum deviation case
    elif max:
        # Masked array case
        if type(newarr)==ma.core.MaskedArray:
            for idx in range(len(newarr)):
                if len(newarr[idx])==0: continue
                if log: newarr[idx] = np.log(idx)
                if not ma.all(newarr[idx].mask) and newarr[idx].std()!=0.0:
                    newarr[idx] = (newarr[idx]-newarr[idx].mean())/np.abs(newarr[idx].std()-newarr[idx].mean())

        # Generic numpy array or list case
        elif type(newarr)==np.ndarray or type(newarr)==list:
            for idx in range(len(newarr)):
                if len(newarr[idx])==0: continue
                if log: newarr[idx] = np.log(idx)
                if np.std(newarr[idx])!=0.0:
                    newarr[idx] = (newarr[idx]-np.mean(newarr[idx]))/np.abs(np.std(newarr[idx])-np.mean(newarr[idx]))

    # Normalize to standard deviation case
    else:
        # Masked array case
        if type(newarr)==ma.core.MaskedArray:
            for idx in range(len(newarr)):
                if len(newarr[idx])==0: continue
                if log: newarr[idx] = np.log(idx)
                if not ma.all(newarr[idx].mask) and newarr[idx].std()!=0.0:
                    newarr[idx] = (newarr[idx]-newarr[idx].mean())/newarr[idx].std()

        # Generic numpy array or list case
        elif type(newarr)==np.ndarray or type(newarr)==list:
            for idx in range(len(newarr)):
                if len(newarr[idx])==0: continue
                if log: newarr[idx] = np.log(idx)
                if np.std(newarr[idx])!=0.0:
                    newarr[idx] = (newarr[idx]-np.mean(newarr[idx]))/np.std(newarr[idx])

    return newarr

def getRingGraph(nNodes,idcs=None):
    """
    Parameters
    ----------
    nNodes : int, required
        Number of nodes in graph
    idcs : list, optional
        List of specific indices to use

    Description
    -----------
    Generates a ring graph structure from a given number 
    of nodes or a list of specific indices.
    """
    l1 = idcs if idcs is not None else [k for k in range(nNodes)]
    l2 = l1[1:]
    l2.append(l1[0])

    # Get directional graph
    graph = dgl.graph((l1,l2))

def getWebGraph(nNodes,idcs=None):
    """
    Parameters
    ----------
    nNodes : int, required
        Number of nodes in graph
    idcs : list, optional
        List of specific indices to use

    Description
    -----------
    Generates a fully connected graph structure from a given number 
    of nodes or a list of specific indices.
    """

    # Generate fully connected graph
    l1 = ak.flatten([[el for el in range(k+1,nNodes)] for k in range(nNodes)])
    l2 = ak.flatten([[k for el in range(k+1,nNodes)] for k in range(nNodes)])

    # Replace indices if requested
    if idcs is not None:
        l1 = [idcs[el] for el in l1]
        l2 = [idcs[el] for el in l2]

    # Get directional graph
    graph = dgl.graph((l1,l2))

#------------------------- Classes: -------------------------#
# Preprocessor, PreprocessorIterator, Constructor

class Constructor:
    """
    Description
    -----------
    ...

    Attributes
    ----------
    ...

    Methods
    -------
    getDataTensor
    setConstruct
    getConstruct
    getGraphs
    """
    def __init__(self,construct=None):
        self.construct = construct

    def getDataTensor(batch,keys): #TODO: Put in constructor object
        """
        Parameters
        ----------
        batch : dict, required
            Dictionary of entry keys to batch data arrays
        keys : list, required
            List of batch keys to use

        Description
        -----------
        Reshapes batch data for specified keys from a dictionary of 
        dimension (nKeys,nEvents,nParticles) to a numpy array of dimension
        (nEvents,nParticles,nKeys).
        """
        return np.moveaxis(np.array([batch[key] for key in keys]),(0,1,2),(2,0,1))

    def setConstruct(construct):
        """
        Parameters
        ----------
        construct : callable, required

        Description
        -----------
        Set graph construct function which should have parameters nNodes and data
        where nNodes is the number of unmasked entries in data and data is the
        data tensor of dimension (nNodes,nFeatures) for the graph.
        """
        self.construct = construct

    def getConstruct():
        """
        Returns
        -------
        construct : callable, required

        Description
        -----------
        Returns graph construct function which should have parameters nNodes and data
        where nNodes is the number of unmasked entries in data and data is the
        data tensor of dimension (nNodes,nFeatures) for the graph.
        """
        return self.construct
        
    def getGraphs(datatensor):
        """
        Parameters
        ----------
        datatensor : numpy.ma.array, required
            Masked tensor array of dimension (nEvents,nNodes,nFeatures)

        Returns
        -------
        List of dgl.graph objects from events in datatensor

        Description
        -----------
        Creates dgl graphs from a events in a data tensor using
        a given graph construction method.
        """
        graphs = []
        for event in datatensor:
            count = ma.count(event)
            if count<=0: continue
            graph = self.construct(count,event)
            graphs.append(graph)
            
        return graphs

class Preprocessor:
    """
    Description
    -----------
    Basic preprocessor object for streamlining preprocessing workflow

    Attributes
    ----------
    file_type
    branches
    labels
    processes

    Methods
    -------
    setFiletype
    getFileType
    addLabels
    setLabels
    getLabels
    addProcesses
    setProcesses
    getProcesses
    process
    setIterArgs
    getIterArgs
    """

    def __init__(self,file_type="root",branches={},labels={},processes={}):
        """
        Parameters
        ----------
        file_type : str, optional
            String indicating preprocessor expected file type
        labels : dict, optional
            Dictionary of branch or bank name to label function
        processes : dict, optional
            Dictionary of branch or bank name to preprocessing function
        """
        self.file_type  = file_type
        self.branches   = branches
        self.labels     = labels
        self.processes  = processes
        self.iterargs   = ()
        self.iterkwargs = {}

    def setFiletype(self, file_type):
        """
        Parameters
        ----------
        file_type : string, required
            String indicating preprocessor expected file type

        Description
        -----------
        Sets file type (either "root" or "hipo") the preprocessor will expect.
        """
        self.file_type = file_type

    def getFileType(self):
        """
        Returns
        -------
        self.file_type : str
            String indicating preprocessor expected file type

        Description
        -----------
        Returns file type (either "root" or "hipo") the preprocessor will expect.
        """
        return self.file_type

    def addBranches(self, branches):
        """
        Parameters
        ----------
        branches : dict, required
            Dictionary of branch/bank names to new branch functions

        Description
        -----------
        Adds dictionary of branch/bank names to functions to produce
        branch entries from existing branches in batch
        """
        for name in branches:
            self.branches[name] = branches[name]

    def setBranches(self, branches):
        """
        Parameters
        ----------
        branches : dict, required
            Dictionary of branch/bank names to new branch functions

        Description
        -----------
        Sets dictionary of branch/bank names to new branch functions
        """
        self.branches = branches

    def getBranches(self):
        """
        Returns
        -------
        self.branches : dict
            Dictionary of branch/bank names to new branch functions

        Description
        -----------
        Returns dictionary of branch/bank names to new branch functions
        """
        return self.branches

    def addLabels(self, labels):
        """
        Parameters
        ----------
        labels : dict, required
            Dictionary of branch/bank names to preprocesssing functions

        Description
        -----------
        Adds dictionary of branch/bank names to preprocessing functions to existing
        dictionary of preprocessing functions
        """
        for name in labels:
            self.labels[name] = labels[name]

    def setLabels(self, labels):
        """
        Parameters
        ----------
        labels : dict, required
            Dictionary of branch/bank names to labelling functions

        Description
        -----------
        Sets dictionary of branch/bank names to labelling functions
        """
        self.labels = labels

    def getLabels(self):
        """
        Returns
        -------
        self.labels : dict
            Dictionary of branch/bank names to labelling functions

        Description
        -----------
        Returns dictionary of branch/bank names to labelling functions
        """
        return self.labels

    def addProcesses(self, processes):
        """
        Parameters
        ----------
        processes : dict, required
            Dictionary of branch/bank names to preprocesssing functions

        Description
        -----------
        Adds dictionary of branch/bank names to preprocessing functions to existing
        dictionary of preprocessing functions
        """
        for name in processes:
            self.processes[name] = processes[name]

    def setProcesses(self, processes):
        """
        Parameters
        ----------
        processes : dict, required
            Dictionary of branch/bank names to preprocesssing functions

        Description
        -----------
        Sets dictionary of branch/bank names to preprocessing functions
        """
        self.processes = processes

    def getProcesses(self):
        """
        Returns
        -------
        self.processes : dict
            Dictionary of branch/bank names to preprocesssing functions

        Description
        -----------
        Returns dictionary of branch/bank names to preprocesssing functions
        """
        return self.processes

    def process(self,batch):
        """
        Parameters
        ----------
        batch : dict, required
            Dictionary of branch/bank names to unprocessed data

        Returns
        -------
        batch : dict
            Dictionary of branch/bank names to preprocessed data

        Description
        -----------
        Applies preprocessing functions to given batch
        """
        for name in self.branches: #NOTE: Define new branches.
            batch[name] = self.branches[name](batch)
        for name in self.labels:   #NOTE: Create labels for classification.
            batch[name] = self.labels[name](batch)
        for name in self.processes: #NOTE: Preprocess data, e.g., by normalization.
            batch[name] = self.processes[name](batch[name])
        return batch

    def setIterArgs(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args
        **kwargs

        Description
        -----------
        Sets iteration arguments and keyword arguments for preprocessor
        """
        self.iterargs   = args
        self.iterkwargs = kwargs

    def getIterArgs(self):
        """
        Returns
        -------
        self.iterags, self.iterkwargs : tuple

        Description
        -----------
        Returns tuple containing iteration arguments and keyword arguments
        for preprocessor
        """
        return (self.iterargs, self.iterkwargs)

    def iterate(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args
        **kwargs
            Parameters for uproot.iterate or hipopy.hipopy.iterate functions

        Returns
        -------
        self : c12gl.preprocessing.Preprocessor
        """
        self.setIterArgs(*args,**kwargs)
        return self

    def __iter__(self):
        """
        Parameters
        ----------
        *args
        **kwargs
            Parameters for uproot.iterate or hipopy.hipopy.iterate functions

        Returns
        -------
        iterator : c12gl.preprocessing.PreprocessorIterator
        """
        return PreprocessorIterator(self, *self.iterargs, **self.iterkwargs)

class PreprocessorIterator:

    """
    Description
    -----------
    Iterator class for c12gl.preprocessing.Preprocessor

    Attributes
    ----------
    preprocessor
    iterator

    Methods
    -------
    ...
    """

    def __init__(self, preprocessor, *args, **kwargs):
        self.preprocessor = preprocessor
        self.iterator     = None
        if self.preprocessor.file_type == "root":
            self.iterator = uproot.iterate(*args, **kwargs).__iter__()
        elif self.preprocessor.file_type == "hipo":
            self.iterator = hipopy.iterate(*args, **kwargs).__iter__()
        else: raise TypeError

    def __next__(self):
        try:
            batch = self.iterator.__next__()
            return self.preprocessor.process(batch)
        except StopIteration: #NOTE: #TODO: Not sure but maybe could just forget the try block since self.iterator.__next__() will already raise the StopIteration exception
            raise StopIteration
        