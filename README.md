# GNN Starter Code

## Step 1: Download

Download this project from your command line by running

`git clone https://github.com/mcembalest/GNN.git`

## Step 2: Enrivonment

Create a new conda environment with python version 3.10:

`conda create --name GNN_ENV python=3.10`

You can now setup the environment for this project by running the included shell script:

`zsh setup_environment.zsh`

Alternatively, you can run the two lines that are in the shell script directly:

`pip install -r requirements.txt`
`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html`

## Step 3: Notebook

Run the notebook `airports.ipynb` to load a Pytorch-Geomtetric graph dataset of US airports. 

Each node is an airport, and each edge indicates the existence of a flight between the airports.

Each node is given a label indicating which quartile of passenger activity the airport has (label==0 means top 25% of passenger activity, label==3 means bottom 25% of passenger activity).

This notebook compares two methods of predicting the activity label of each airport: a baseline logistic regression model vs graph neural networks.
