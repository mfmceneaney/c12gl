#----------------------------------------------------------------------#
# Example dataset creation script
# Author: Matthew McEneaney
# Contact: matthew.mceneaney@duke.edu
#----------------------------------------------------------------------#

# Data Imports
import numpy as np
from numpy import ma
import awkward as ak
import pandas as pd

# Custom Imports
from c12gl.preprocessing import pad, normalize, getRingGraph, getWebGraph, Constructor, Preprocessor
from c12gl.dataloading import loadGraphDataset, GraphDataset

# Random Imports
from tqdm import tqdm

#-------------------- DATASET CREATION --------------------#

# Define graph construction
def construct(nNodes,data):
    g = getWebGraph(nNodes)
    g.ndata['data'] = data
    return g

# Create DGL Dataset
name        = 'test_dataset'
num_classes = 2
ds          = GraphDataset(name=name,num_classes=num_classes)

# Set parameters for first data distribution
filename  = 'file.hipo'
file_type = 'hipo'
banks     = ['REC::Particle','MC::Lund','RUN::config']
step      = 100
items     = ['px', 'py', 'pz','beta']
keys      = [banks[0]+'_'+item for item in items]

# Create preprocessor and constructor
p = Preprocessor(file_type=file_type)
c = Constructor(construct=construct)

# Define labels
def getLabel(batch):
    return [1 if np.any([True if el==3122 and arr[batch['MC::lund_daughter'][arr_idx][el_idx]]==2212 else False for el_idx, el in enumerate(arr)]) else 0 for arr_idx, arr in enumerate(batch['MC::Lund_pid'])]
    #NOTE: #TODO: Assign additional labels...could add get label tensor method below...

# Add labels to assign
label_key = 'ML::Label'
p.addLabels({label_key:getLabel})

# Add processes with kwargs
for item in items:
    kwargs = {}
    p.addProcesses({banks[0]+'_'+item:[normalize,kwargs]})

# Convert to padded numpy.ma arrays (needed for getGraphs below)
for item in items:
    kwargs = {'target_dim':100, 'axis':1}
    p.addProcesses({banks[0]+'_'+item:[pad,kwargs]})

# Loop files and build dataset
for idx, batch in tqdm(enumerate(p.iterate(filename,banks=banks,step=step))):
    ls = batch[label_key]
    datatensor = c.getDataTensor(batch,keys) #NOTE: Can filter datatensor for NaN here.
    gs = c.getGraphs(datatensor) #NOTE: Check what is taking the longest here...
    ds.extend(ls,gs)

print('len(ds) = ',len(ds)) #INFO
