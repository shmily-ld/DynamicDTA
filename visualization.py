import torch
from torch_geometric.loader import DataLoader
from models.gcn import GCNNet
from utils import TestbedDataset
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_class = GCNNet
model = model_class().to(device)

model_file_name = 'model_GCNNet_data.model' 
model.load_state_dict(torch.load(model_file_name))


predict_data = TestbedDataset(root='data', dataset='data_test')  
predict_loader = DataLoader(predict_data, batch_size=512, shuffle=False)


model.eval()
total_preds = torch.Tensor()
total_labels = torch.Tensor()
with torch.no_grad():
    for data in predict_loader:
        data = data.to(device)
        output = model(data)
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)


predictions = total_preds.numpy().flatten()
labels = total_labels.numpy().flatten()


plt.figure(figsize=(8, 8))
plt.scatter(labels, predictions, alpha=0.5,color="#FF8F6B", label="Predicted vs True Affinities")
plt.plot([min(labels) - 1, max(labels) + 1], [min(labels) - 1, max(labels) + 1], 'r--', label="y = x (Ideal)")


plt.xlabel("Affinity (True)", fontsize=14)
plt.ylabel("Affinity (Predicted)", fontsize=14)
plt.title("Predicted Affinity vs True Affinity", fontsize=16)
plt.legend(fontsize=12, frameon=False)
plt.grid(False)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("predicted_vs_true_affinity.svg", format="svg")


plt.show()
