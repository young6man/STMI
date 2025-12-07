import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        path = path.to(X.dtype)
        X = torch.matmul(path, X) 
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)  
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg("mean")
        self.e2v = MessageAgg("mean")

    def forward(self, x, H):
        x = self.fc(x)
        E = self.v2e(x, H.transpose(1, 2).contiguous()) 
        x = self.e2v(E, H) 
        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold=5.0, hidden_size=128):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, d = x.shape
        x = x.transpose(1, 2).contiguous() 
        feature = x.clone()
        distance = torch.cdist(feature, feature) 
        hg = (distance < self.threshold).float().to(x.device) 
        x = self.hgconv(x, hg) + x 
        x = x.transpose(1, 2).contiguous()
        x = self.act(self.norm(x))
        return x
