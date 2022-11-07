## Standard libraries
import os
import json
import math
import numpy as np
import time
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from urllib.request import urlopen
import json
import matplotlib.pyplot as plt

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# NetworkX
import networkx as nx

## PyTorch
import torch

def visualize_graph(ax, G, color=None, labels=False, title=None, layout=lambda G : nx.spring_layout(G, seed=42)):
    ax.set_xticks([])
    ax.set_yticks([])
    node_color = 'white'
    if color is not None: node_color = color
    nx.draw_networkx(G, pos=layout(G), with_labels=labels,
                     node_color=node_color, cmap="Set2", node_size=100, ax=ax)
    if title: ax.set_title(title)

def visualize_embedding(ax, h, color, epoch=None, loss=None, title=''):
    ax.set_xticks([])
    ax.set_yticks([])
    h = h.detach().cpu().numpy()
    ax.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        ax.set_title(title + f'\n Epoch: {epoch}, Loss: {loss.item():.4f}')
    ax.set_xlabel('embedding dim 1')
    ax.set_ylabel('embedding dim 2')
        
def visualize_model_embedding_spaces(models, modelnames):
    fig, ax = plt.subplots(1, len(models), figsize=(len(models)*4,4))
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    for i, model in enumerate(models):
        if len(models)==1:
            _ax = ax
        else:
            _ax = ax[i]
        assert model.classifier.in_features == 2
        name = modelnames[i]
        pred = model.classifier(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()).detach().numpy()
        zz = np.argmax(pred, axis=1).reshape(xx.shape)
        _ax.pcolormesh(xx, yy, zz)
        _ax.set_title(name)
        _ax.set_xlabel('embedding dim 1')
        _ax.set_ylabel('embedding dim 2')
    plt.tight_layout()
    plt.show()    
        
def plot_training_results(losses, acc, modelnames):
    assert list(losses.keys()) == modelnames and list(acc.keys()) == modelnames
    n_models = len(modelnames)
    fig, ax = plt.subplots(2, n_models, figsize=(3*n_models,6))
    if n_models == 1:
        model_acc = acc[modelnames[0]]
        ax[0].set_title(f'{modelnames[0]}: accuracy')
        ax[0].plot(np.arange(len(model_acc)), [x[0] for x in model_acc], label='train acc')
        ax[0].plot(np.arange(len(model_acc)), [x[1] for x in model_acc], label='test acc')
        ax[0].set_xlabel('# epochs')
        ax[0].legend(loc='lower right')
        ax[0].set_ylim(([0,1]))

        model_losses = losses[modelnames[0]]
        ax[1].set_title(f'{modelnames[0]}: loss')
        ax[1].plot(np.arange(len(model_losses)), model_losses)
        ax[1].set_xlabel('# epochs')
        ax[1].set_ylim(([0,max([max(losses[name]) for name in modelnames])]))
    else:
        for i in range(n_models):
            model_acc = acc[modelnames[i]]
            ax[0,i].set_title(f'{modelnames[i]}: accuracy')
            ax[0,i].plot(np.arange(len(model_acc)), [x[0] for x in model_acc], label='train acc')
            ax[0,i].plot(np.arange(len(model_acc)), [x[1] for x in model_acc], label='test acc')
            ax[0,i].set_xlabel('# epochs')
            ax[0,i].legend(loc='lower right')
            ax[0,i].set_ylim(([0,1]))

            model_losses = losses[modelnames[i]]
            ax[1,i].set_title(f'{modelnames[i]}: loss')
            ax[1,i].plot(np.arange(len(model_losses)), model_losses)
            ax[1,i].set_xlabel('# epochs')
            ax[1,i].set_ylim(([0,max([max(losses[name]) for name in modelnames])]))
    plt.tight_layout()
    
def plot_graph_geography(graph, geodf, title=None):
    
    # gets latitude & longitude coords for each node by ID from geodf
    geo_df_row = lambda i : geodf.loc[geodf.index==i]
    
    # gets edge connections from graph of nodes in geodf
    connections = graph.subgraph(geodf.index).edges()
    
    node_ids = np.array(connections).flatten()
    
    connections_coords = np.array([[geo_df_row(i).lon, geo_df_row(i).lat] for i in node_ids]).reshape(len(node_ids), 2)

    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon = connections_coords[:,0],
                lat = connections_coords[:,1],
                mode = 'lines',
                line=dict(
                    color='black', 
                    width=.03)
            ),
            go.Scattergeo(
                lon = geodf['lon'],
                lat = geodf['lat'],
                text = geodf['name'],
                mode = 'markers',
                hoverinfo='text'
            )
    ])  

    fig.update_geos(fitbounds='locations')
    if title: fig.update_layout(title=title, showlegend=False)
    fig.show()

