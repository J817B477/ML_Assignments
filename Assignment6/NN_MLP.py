import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, feat_in, feat_out, dropout=0.1):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(feat_in, 2*feat_in)
        self.bn1 = nn.BatchNorm1d(2*feat_in)

        self.fc2 = nn.Linear(2*feat_in, 3*feat_in)
        self.bn2 = nn.BatchNorm1d(3*feat_in)
        
        self.fc3 = nn.Linear(3*feat_in, 5*feat_in)
        self.bn3 = nn.BatchNorm1d(5*feat_in)

        self.fc4 = nn.Linear(5*feat_in, 3*feat_in)
        self.bn4 = nn.BatchNorm1d(3*feat_in)

        self.fc5 = nn.Linear(3*feat_in, 2*feat_in)
        self.bn5 = nn.BatchNorm1d(2*feat_in)

        self.fc6 = nn.Linear(2*feat_in, feat_out)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()



    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))

        x = self.fc2(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(self.bn3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(self.bn4(x)) 
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.relu(self.bn5(x)) 
        x = self.dropout(x)

        x = self.fc6(x)

        return x

if __name__ == "__main__":
    batch_size = 8
    feat_in = 20

    fake_input = torch.randn(
        [batch_size, feat_in]
    )
    print(fake_input.shape)

    mlp = MLP(
        feat_in=feat_in, 
        feat_out=4
    )
    print(mlp)

    res = mlp(fake_input)
    print(res.shape)    # [batch_size, feat_out]