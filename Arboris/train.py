"""
Arboris - Model (Final Version)

Purpose:
- Custom CNN backbone
- Transformer head
- Multi-head classification
"""

from imports import *
from paths import *

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class TransformerHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x.squeeze(1)


class Model(nn.Module):
    def __init__(self, tax_sizes):
        super().__init__()

        self.cnn = CNN()
        self.transformer = TransformerHead(128)

        self.heads = nn.ModuleDict({
            lvl: nn.Linear(128, tax_sizes[lvl])
            for lvl in TAXONOMY_LEVELS
        })

    def forward(self, x):
        x = self.cnn(x)
        x = self.transformer(x)

        return {lvl: self.heads[lvl](x) for lvl in TAXONOMY_LEVELS}


def loss_fn(outputs, targets):
    loss = 0
    for i, lvl in enumerate(TAXONOMY_LEVELS):
        loss += F.cross_entropy(outputs[lvl], targets[:, i])
    return loss