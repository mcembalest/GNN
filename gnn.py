import numpy as np
import torch
from torch.nn import Linear

loss_fn = torch.nn.CrossEntropyLoss()


class GNN(torch.nn.Module):
    def __init__(self, layer, num_nodes, num_classes):
        """
        Args:
            layer: pytorch-geometric graph convolution
            num_nodes: int, number of nodes in the graph
            num_classes: int, number of possible classes to assign to each node
        """
        super().__init__()
        self.conv1 = layer(num_nodes, 50)
        self.conv2 = layer(50, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        """
        Args:
            x: input features of each node
            edge_index: sparse representation of edges between nodes (shape: 2 x num_edges)
        Returns:
            out: classifications of each node
            h: embedding of each node
        """
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h
    

def train_step(model, optimizer, data, train_mask):
    """GNN training step

    Args:
        model: GNN
        optimizer: torch optimizer
        data: torch_geometric.data.data.Data
        train_mask: (n_nodes,) nd.array
    Returns:
        loss: float, loss value for the train step
        out: node predictions
        h: node embeddings
    """
    model.train()
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = loss_fn(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.detach().numpy(), out, h


def masked_accuracy(data, out, mask, verbose=True):
    """Measures graph node classification accuracy with a mask

    Args:
        data: torch_geometric.data.data.Data
        out: nd.array
            predicted labels for each node in a graph
        mask: nd.array
            indicator for which nodes we are calculating accuracy for (train or test)
    Returns:
        acc: float, % of pred labels matching data labels
    """
    pred = out[mask].argmax(dim=1)
    acc = int((pred==data.y[mask]).sum()) / mask.sum()
    return acc


def train(
    models, 
    modelnames, 
    data, 
    train_mask, 
    test_mask, 
    learning_rate=1e-3,
    n_train_steps=1000, 
    n_per_eval=100,
    seed=64
):
    """Train GNNs

    Args:
        models: list of pytorch GNN models
        modelnames: list[str] of model names
        data: torch_geometric.data.data.Data
        train_mask: nd.array
            indicator for which nodes are in the train set
        test_mask: nd.array
            indicator for which nodes are in the test set
    Returns:
        metrics: dict[metric_name : dict[model_name : list[union[float, list[float]]]]]
            evaluating model metrics & last-layer embeddings every n_per_eval steps
    """
    torch.manual_seed(seed)
    metrics = {'loss' : {m : [] for m in modelnames}, 
               'accuracy' : {m : [] for m in modelnames}, 
               'embeddings' : {m : [] for m in modelnames}}

    for i, (model, name) in enumerate(zip(models, modelnames)):
        print(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for step in range(n_train_steps):
            loss, out, h = train_step(model, optimizer, data, train_mask)
            if step % n_per_eval == 0:
                train_acc = masked_accuracy(data, out, train_mask)
                test_acc = masked_accuracy(data, out, test_mask)
                metrics['loss'][name].append(loss.item())
                metrics['accuracy'][name].append((train_acc, test_acc))
                metrics['embeddings'][name].append(h.tolist())
                print(f'[{step} iter.] loss: {np.around(loss, 2)}, test acc: {np.around(test_acc, 2)}')
    return metrics
