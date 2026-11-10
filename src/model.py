"""Compact CNN for 14-label ChestMNIST classification (99,438 parameters).

Architecture exactly as registered in the manuscript (Table 4):
  Block 1: 3x3 conv 1->16,  BatchNorm, ReLU, 2x2 MaxPool   (192 params)
  Block 2: 3x3 conv 16->32, BatchNorm, ReLU, 2x2 MaxPool   (4,704 params)
  Block 3: 3x3 conv 32->64, BatchNorm, ReLU, 2x2 MaxPool   (18,624 params)
  Block 4: 3x3 conv 64->128, BatchNorm, ReLU               (74,112 params)
  Adaptive average pooling -> 128, Dropout 0.2, Linear 128->14 (1,806 params)
"""
import torch
import torch.nn as nn


class CompactCNN(nn.Module):
    def __init__(self, n_labels: int = 14, dropout: float = 0.2):
        super().__init__()

        def block(cin, cout, pool):
            layers = [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 16, pool=True),
            block(16, 32, pool=True),
            block(32, 64, pool=True),
            block(64, 128, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)  # logits; apply sigmoid outside


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CompactCNN()
    n = count_parameters(m)
    print(f"trainable parameters: {n}")
    assert n == 99438, f"expected 99,438 parameters, got {n}"
    y = m(torch.zeros(2, 1, 64, 64))
    assert y.shape == (2, 14)
    print("architecture check passed")
