# GNN Starter Code

## Step 1: 

Download this project from your command line by running

`git clone https://github.com/mcembalest/GNN.git`

## Step 2: 

Setup your conda environment for this project by running the included shell script:

`zsh setup_environment.zsh`

## Step 3: 

Run the notebook `airports.ipynb` to load a Pytorch-Geomtetric graph dataset of US airports. Each node is an airport, each edge indicates the existence of a flight between the airports, and each node is given a label indicating which quartile of passenger activity the airport has (label==0 means top 25% of passenger activity, label==3 means bottom 25% of passenger activity)

This notebook compares a baseline logistic regression model vs two different types of graph neural networks.
