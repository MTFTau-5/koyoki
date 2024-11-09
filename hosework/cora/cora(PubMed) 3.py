#这个模型是ai缝的，效果并不好，，我昨晚看了一个cnn+gcn的论文有感（但是碍于实力只能ai救一下了）
#经过测试，cnn+gcn对于复杂数据的处理能力更强（比较cora与pubmed），当然不知道为啥正确率比不上单独用gcn的
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载并预处理Cora数据集
dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=NormalizeFeatures())

class CNNforCora(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CNNforCora, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 数据集信息
data = dataset[0].to(device)  
input_dim = dataset.num_node_features
hidden_dim = 16  
output_dim = dataset.num_classes

# 初始化模型
cnn_model = CNNforCora(input_dim, hidden_dim).to(device)  
gcn_model = GCN(hidden_dim, hidden_dim, output_dim).to(device)  
# 组合模型
class CombinedModel(nn.Module):
    def __init__(self, cnn_model, gcn_model):
        super(CombinedModel, self).__init__()
        #self.cnn_model = cnn_model
        self.gcn_model = gcn_model
        self.cnn_model = cnn_model

    def forward(self, x, edge_index):
        x = self.cnn_model(x)
        x = self.gcn_model(x, edge_index)
        return F.log_softmax(x, dim=1)

combined_model = CombinedModel(cnn_model, gcn_model).to(device)
#optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001, weight_decay=5e-4)  
#optimizer = AdamW(combined_model.parameters(), lr=0.001, weight_decay=5e-4)
#optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adagrad(combined_model.parameters(), lr=0.001)
optimizer = torch.optim.RMSprop(combined_model.parameters(), lr=0.001)
#因为结果不理想，所以找了多组优化器看看哪个最好
#目前结果是在100Epoch后，RMSprop最佳，loss值下降较快，其次是SGD但是loss较高，其他的估计在一万次之后只能维持在百分之50多的正确率
#而RMS 100次便可达到百分之40多，SGD为30多，后续还没测试

# 训练模型
combined_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = combined_model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 测试模型
combined_model.eval()
_, pred = combined_model(data.x, data.edge_index).max(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {accuracy:.4f}')
