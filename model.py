"""Model definition and shared constants.

Separated from training and serving so both load the exact same
architecture — avoids the classic bug where the server rebuilds a
subtly different model than the one that was trained.
"""

import torch.nn as nn

FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisNet(nn.Module):
    """A small 2-layer MLP: 4 features -> 16 hidden -> 3 classes."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)
