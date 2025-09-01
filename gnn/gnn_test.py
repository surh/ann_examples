import os
import glob
import torch
import pandas as pd
import numpy as np
# from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Dataset
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx

class dataset_loader(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.graph_files = sorted(glob.glob(os.path.join(root_dir, '*_adj.csv')))

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        adj_path = self.graph_files[idx]
        base_name = os.path.basename(adj_path).replace('_adj.csv', '')
        target_path = os.path.join(self.root_dir, f"{base_name}_targets.csv")
        feature_path = os.path.join(self.root_dir, f"{base_name}_features.csv")

        adj = pd.read_csv(adj_path, header=None).values
        target = pd.read_csv(target_path, header=None).values.squeeze()

        assert adj.shape[0] == adj.shape[1] == len(target), "Mismatch between adjacency and targets"

        row, col = np.nonzero(adj)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_weight = torch.tensor(adj[row, col], dtype=torch.float32)

        # Handles optional node level features
        if os.path.exists(feature_path):
            x = pd.read_csv(feature_path, header=None).values
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = torch.ones((adj.shape[0], 1))

        y = torch.tensor(target, dtype=torch.float32)

        if torch.isnan(y).any():
            raise ValueError(f"NaN target values in {base_name}")
        if torch.isnan(edge_weight).any():
            raise ValueError(f"NaN edge weights in {base_name}")
        if torch.isnan(x).any():
            raise ValueError(f"NaN node features in {base_name}")

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)


class simple_gnn_gcn(nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.conv1 = GraphConv(1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x.view(-1)  # [num_nodes]


# Prepare data
input_data = dataset_loader("dummy_data")
# input_data[0]['x']
# input_data[0]['edge_index']
# input_data[0]['edge_attr']
input_data_loader = DataLoader(input_data, batch_size=1, shuffle=True) 

# Train model
gnn_model = simple_gnn_gcn()

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
for epoch in range(100):
    total_loss = 0
    for batch in input_data_loader:
        optimizer.zero_grad()
        out = gnn_model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")



# Plot
data = input_data[0]
gnn_model.eval()
with torch.no_grad():
    preds = gnn_model(data)

true_vals = data.y.numpy()
pred_vals = preds.numpy()

plt.figure(figsize=(6,6))
plt.scatter(true_vals, pred_vals, alpha=0.7)
plt.plot([true_vals.min(), true_vals.max()],
         [true_vals.min(), true_vals.max()],
         'r--')  # ideal line
plt.xlabel("True Node Values")
plt.ylabel("Predicted Node Values")
plt.title("GraphConv Predictions vs Targets")
plt.show()




# Convert to NetworkX
edge_index = data.edge_index.numpy()
G = nx.Graph()
G.add_edges_from(edge_index.T)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
nx.draw_networkx(G, node_color=true_vals, cmap="viridis", with_labels=False)
plt.title("True Node Values")

plt.subplot(1,2,2)
nx.draw_networkx(G, node_color=pred_vals, cmap="viridis", with_labels=False)
plt.title("Predicted Node Values")

plt.show()