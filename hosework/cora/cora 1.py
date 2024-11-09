import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# 加载并预处理Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)  
        #self.fc2 = torch.nn.Linear(hidden_dim, output_dim)  
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)  # 展平图结构数据以适应全连接层
        #x = F.relu(self.fc1(x))  
        #x = self.fc2(x) 

        return F.log_softmax(x, dim=1)

# 数据集信息
data = dataset[0]
input_dim = dataset.num_node_features
hidden_dim = 100
output_dim = dataset.num_classes

# 初始化模型
model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# 训练模型
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 测试模型
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')


