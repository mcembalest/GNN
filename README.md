# GNN Starter Code

Run notebook `airports.ipynb` to view an example graph dataset and train graph neural networks to model the data (comparing against a baseline logistic regression model)

## Airports Data

Graphs like the map of US airports contain useful information when modeling data with a network shape:

![Airports dataset](img/airports.png "Airports dataset")

Each node in the graph is an airport, and each edge indicates the existence of a flight between the airports.

Each airport in this dataset has a label representing its activity level (AKA how many passengers the airport gets each year, relative to other airports. The label 0 means top 25% of passenger activity, and the label 3 means bottom 25% of passenger activity, etc).

## Graph Neural Network

Our graph neural network encodes each airport in the network into a 2d vector, which is then used to classify the network's activity level. We show how to visualize the embeddings of the final layer during training to qualitatively evaluate how the models learn to classify the data.

![GNN Embeddings](img/trajectories.gif "Trajectory of GNN 2d node embeddings during training")

## Setup

Download this project from your command line by running

`git clone https://github.com/mcembalest/GNN.git`

Create a new conda environment with python version 3.10:

`conda create -n <envname> python=3.10`

You can now setup the environment for this project by running the included shell script:

1. Activate the new environment:

    `conda activate <envname>`

2. Install libraries into the new environment:

    `zsh setup_environment.zsh`

3. If step 2 worked, ignore step 3. If step 2 did not work, you can run the two lines that are in `setup_environment.zsh` directly:

    `pip install -r requirements.txt`

    `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html`
