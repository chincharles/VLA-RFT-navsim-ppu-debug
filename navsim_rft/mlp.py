"""timm Mlp-compatible primitive; avoids importing timm for CPU core tests."""
from torch import nn
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features or in_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features or in_features, out_features or in_features)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
