#-------------------- DATASET CREATION --------------------#
import sys
sys.path.append('..')
from c12gl.preprocessing import *
from c12gl.dataloading import *

# Define graph construction
def construct(nNodes,data):
    g = getWebGraph(nNodes)
    g.ndata['data'] = data
    return g

# Create DGL Dataset
name        = "test_dataset_10_17_22"
num_classes = 2
ds          = GraphDataset(name=name,num_classes=num_classes)

# Set parameters for first data distribution
filename = "data1.hipo"
bank     = "NEW::bank"
step     = 100
items    = ["px", "py", "pz"]
keys     = [bank+"_"+item for item in items]

# Create preprocessor and constructor
p = Preprocessor(file_type="hipo")
c = Constructor(construct=construct)

# Define labels
def getLabel(batch):
    return [1 for arr in batch[bank+"_px"]]

# Add labels to assign
label_key = "ML::Label"
p.addLabels({label_key:getLabel})

# # Add processes with kwargs
# for item in items:
#     kwargs = {}
#     p.addProcesses({bank+"_"+item:[normalize,kwargs]})

# Loop files and build dataset
for idx, batch in enumerate(p.iterate(filename,banks=[bank],step=step)):
    ls = batch[label_key]
    datatensor = c.getDataTensor(batch,keys) #NOTE: Can filter datatensor for NaN here.
    gs = c.getGraphs(datatensor) #NOTE: Check what is taking the longest here...
    ds.extend(ls,gs)

print("len(ds) = ",len(ds))