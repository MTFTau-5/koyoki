import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import math
from sklearn.model_selection import train_test_split


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class AudioDataset(Dataset):
    def __init__(self, pkl_file_path, num_devices):
        with open(pkl_file_path, 'rb') as otto:
            all_data = pickle.load(otto)
        self.merged_data = []
        for snr_data in all_data:
            self.merged_data.extend(snr_data)
        self.num_devices = num_devices
        
        self.embedding_model = EmbeddingModel(num_devices)

    def __len__(self):
        return len(self.merged_data)



    def __getitem__(self, index):
        mfcc_feature = self.merged_data[index][0]
        device_num = self.merged_data[index][1]
        label = self.merged_data[index][2]
        mfcc_feature_tensor = torch.from_numpy(mfcc_feature).float()
        device_num_tensor = torch.tensor(device_num).long()
        
        device_num_embedding = self.embedding_model(device_num_tensor)
        
        return mfcc_feature_tensor, device_num_embedding, label





class EmbeddingModel(nn.Module):
    def __init__(self,num_devices):
        super(EmbeddingModel,self).__init__()
        self.device_num = nn.Embedding(num_devices, 11)

    def forward(self,device_num_tensor):
        
        device_num_embedding = self.device_num(device_num_tensor)
        return device_num_embedding





class CNN(nn.Module):
    def __init__(self, num_classes1):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(13, 32, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        self.fc = nn.Linear(768, num_classes1)


    def forward(self, x):
        x = x.squeeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        print("head_dim:", self.head_dim)
        print("num_heads:", self.num_heads)
        print("embed_dim:", self.embed_dim)
        assert self.head_dim * num_heads == self.embed_dim


        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, self.num_heads, 3 * self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)


        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, src):
        src2 = self.self_attn(src)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])


    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


# 定义包含 Transformer 的音频分类模型
class AudioTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=8, dim_feedforward=1024):
        super(AudioTransformerModel, self).__init__()
        self.cnn = CNN(416) 
        encoder_layer = TransformerEncoderLayer(416, num_heads, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(416, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.cnn(x)  
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1).contiguous()
        x = nn.MaxPool2d(kernel_size=1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    pkl_file_path = '/home/mtftau-5/workplace/dataset/data.pkl'
    num_devices = 10 
    audio_dataset = AudioDataset(pkl_file_path, num_devices)
    train_data, test_data = train_test_split(audio_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


    input_dim = 153
    num_classes = 2


    model = AudioTransformerModel(input_dim, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0030)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    num_epochs = 40
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_features, batch_device_nums, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            #print(f"Output matrix size: {outputs.size()}")
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_device_nums, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    print(f"Final test loss: {test_loss / len(test_loader)}")
    print(f"Final test accuracy: {100 * correct / total}%")
