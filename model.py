import torch
from torch import nn

class CartPoleModel(nn.Module):
    def __init__(self, hidden: int):
        super(CartPoleModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
